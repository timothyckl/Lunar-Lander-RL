import os
import wandb
import numpy as np 
from time import time
from random import random
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, PrioritizedBuffer, EpisodeSaver


class DQN:
    def __init__(self, env, discount, learning_rate, exploration, exploration_decay, exploration_min=0.01, target_update_interval=100, with_per=False, log_wandb=False):
        self.env = env
        self.state_size = env.observation_space.shape[0]  # number of possible states
        self.action_size = env.action_space.n  # number of possible actions
        self.discount = discount  # discount rate
        self.learning_rate = learning_rate  # learning rate
        self.exploration = exploration  # exploration rate
        self.exploration_decay = exploration_decay  # exploration decay rate
        self.exploration_min = exploration_min  # minimum exploration rate
        self.with_per = with_per  # prioritized experience replay

        if with_per:
            self.memory = PrioritizedBuffer()

        self.memory = ReplayBuffer(max_length=10_000)
        self.batch_size = 64
        self.max_steps = 2000

        # we use two networks: one for training and one for target
        # this is to stabilize the Q-learning algorithm
        # source: https://ai.stackexchange.com/questions/6982/why-does-dqn-require-two-different-networks
        self.qnet_local = self.create_qnet(name='qnet_local')
        self.qnet_target = self.create_qnet(name='qnet_target')
        self.target_update_interval = target_update_interval

        self.log_wandb = log_wandb

    def create_qnet(self, name):
        model = Sequential([
            Dense(units=256, activation='relu', input_shape=(self.state_size,)),
            Dense(units=256, activation='relu'),
            Dense(units=256, activation='relu'),
            Dense(units=self.action_size, activation='linear')  # linear activation for Q(s, a)
        ], name=name)

        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))

        return model
    
    def act(self, state):
        '''
        Epsilon-greedy policy is used to choose action.
        This means that if we choose to exploit, we choose the action with the highest Q-value.
        '''
        if random() > self.exploration:
            # exploit
            return np.argmax(self.qnet_local.predict_on_batch(state))
        else:
            # explore
            return self.env.action_space.sample()

    def update_local(self):
        '''
        Q(S, A) ← Q(S, A) + α[R + γmax_a'Q(S', a) - Q(S, A)]

        where 
        - Q(S, A) is being updated
        - Q(s, a) is the current Q-value
        - R + γmax_a'Q(S', a) is the target Q-value
        '''
        # sample randomly from memory to prevent correlation
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        targets = reward + self.discount * np.amax(self.qnet_target.predict_on_batch(next_state), axis=1) * (1 - done)  
        target_vector = self.qnet_local.predict_on_batch(state)
        idx = np.array([i for i in range(self.batch_size)])
        target_vector[[idx], [action]] = targets

        self.qnet_local.fit(state, target_vector, epochs=1, verbose=0)  # update Q(s, a)

    def update_target(self):
        '''
        Update target network's weights with local network's weights
        source: https://ai.stackexchange.com/questions/21984/in-deep-q-learning-are-the-target-update-frequency-and-the-batch-training-frequ
        '''
        self.qnet_target.set_weights(self.qnet_local.get_weights())

    def train(self, n_episodes, update_qnets=True):
        '''
        ---------------------------------------------------------
        Deep Q-Learning with Experience Replay
        ---------------------------------------------------------
        Initialize replay memory D to capacity N
        Initialize action-value function Q with random weights

        for episode = 1, M do
            Initialize sequence s1 = {x1} and preprocessed sequence φ1 = φ(s1)
            for t = 1, T do
                With probability ε select a random action at
                otherwise select at = argmaxaQ(φ(st), a; θ)
                Execute action at in emulator and observe reward rt and new state st+1
                Set st+1 = st, at and preprocess φt+1 = φ(st+1)
                Store transition (φt, at, rt, φt+1) in D
                Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
                Set yj = rj if the episode ends at j + 1, otherwise yj = rj + γmaxaQ(φj+1, a; θ)
                Perform a gradient descent step on (yj − Q(φj , aj ; θ))2 with respect to the network parameters θ
            end for
        end for
        '''
        for episode in range(n_episodes):
            state = self.env.reset()
            state = state[0].reshape(1, self.state_size)
            done = False
            episode_reward = 0
            episode_steps = 0
            frames = []
            rewards_list = []
            exploration_rate_list = []
            steps_per_episode_list = []
            # tqdm_e = tqdm(range(self.max_steps), desc='Episode {}/{}'.format(episode, n_episodes), leave=False, unit='step')
            start_time = time()

            for step in range(self.max_steps):
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.state_size)
                frames.append(self.env.render())

                if update_qnets:
                    self.memory.append((state, action, reward, next_state, done))

                    # if memory is large enough, update local qnet Q(S, A)
                    if len(self.memory) > self.batch_size:
                        self.update_local()

                    # update target qnet to match local qnet 
                    if step % self.target_update_interval == 0:
                        self.update_target()

                # update state
                state = next_state
                episode_reward += reward
                episode_steps += 1

                rewards_list.append(episode_reward)
                exploration_rate_list.append(self.exploration)
                steps_per_episode_list.append(episode_steps)

                if done:
                    break

            print(f'[EP {episode+1}/{n_episodes}]  Rewards: {episode_reward:.4f} | Steps: {episode_steps:.0f} | Eps: {self.exploration:.4f} | Time: {time() - start_time:.4f}s')

            # update exploration rate
            self.exploration = max(self.exploration_min, self.exploration * self.exploration_decay)
            
            # save the last episode as a gif every 10 episodes
            if ((episode + 1) % 100 == 0) or (episode == 0):
                saver = EpisodeSaver(self.env, frames, algo='DQN', episode_number=episode + 1)
                saver.save()

            if self.log_wandb:
                wandb.log({
                    'rewards': episode_reward,
                    'steps': episode_steps,
                    'exploration': self.exploration
                })

        self.env.close()
        wandb.finish()

        return rewards_list, exploration_rate_list, steps_per_episode_list

    def save(self, name):
        if os.path.exists('assets'):
            os.makedirs('assets')

        self.qnet_local.save(f'assets/{name}')