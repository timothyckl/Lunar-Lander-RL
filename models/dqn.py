import os
import wandb
import numpy as np
from time import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver


class DQN:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=0.99, epsilon_min=0.01, 
                 batch_size=64, random_engine_fail=False, engine_fail_prob=0.5, fname='DQL'):
        self.env = env 
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.state_size = self.env.observation_space.shape[0]
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.random_engine_fail = random_engine_fail
        self.engine_fail_prob = engine_fail_prob
        self.fname = fname
        self.memory = ReplayBuffer(10_000, self.state_size, self.action_size)
        self.qnet = self.create_qnet('qnet')

    def create_qnet(self, name):
        model = Sequential([
            Dense(units=256, activation='relu', input_shape=(self.state_size,)),
            Dense(units=256, activation='relu'),
            Dense(units=self.action_size, activation='linear')
        ], name=name)

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)

    def act(self, state):
        '''
        Epsilon-greedy policy is used to choose action.
        This means that if we choose to exploit, we choose the action with the highest Q-value.
        '''
        state = np.reshape(state, [1, self.state_size])
        act_rand = np.random.random()  # random number to determine if we should explore or exploit

        if act_rand < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.qnet.predict(state, verbose=0))

    def update(self):
        '''
        Q(S, A) ← Q(S, A) + α[R + γmax_a'Q(S', a) - Q(S, A)]

        where 
        - Q(S, A) is being updated
        - Q(s, a) is the current Q-value
        - R + γmax_a'Q(S', a) is the target Q-value
        '''
        if self.memory.memory_counter > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_idx = np.dot(action, action_values)

            q_current = self.qnet.predict(state, verbose=0)
            q_future = self.qnet.predict(new_state, verbose=0)
            q_target = q_current.copy()

            batch_idx = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_idx, action_idx] = reward + self.gamma * np.max(q_future, axis=1) * done

            self.qnet.fit(x=state, y=q_target, verbose=0)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, n_episodes, max_steps=1000, log_wandb=False, 
                update=True, save_episodes=False, save_interval=10):
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
        history = {'reward': [], 'avg_reward_100': [], 'steps': []}

        for episode in range(n_episodes):
            start_time = time()
            state = self.env.reset()
            state = state[0]
            done = False
            episode_reward = 0
            episode_steps = 0
            frames = []

            for _ in range(max_steps):
                # if self.random_engine_fail is true, then there is 
                # a self.engine_fail_prob chance that the engine will fail
                # else, the agent will act as normal
                if self.random_engine_fail:
                    if np.random.random() < self.engine_fail_prob:
                        action = 0
                    else:
                        action = self.act(state)
                else:
                    action = self.act(state)
                
                new_state, reward, done, _, _ = self.env.step(action)
                frames.append(self.env.render())

                if update:
                    self.remember(state, action, reward, new_state, done)
                    self.update()

                state = new_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if log_wandb:
                wandb.log({
                    'reward': episode_reward,
                    'steps': episode_steps, 
                    'epsilon': self.epsilon
                })

            if save_episodes:
                if (episode + 1) % save_interval == 0 or (episode == 0):
                    s = EpisodeSaver(self.env, frames, self.fname, episode + 1)
                    s.save()

            print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

            history['reward'].append(episode_reward)
            history['avg_reward_100'].append(np.mean(history['reward'][-100:]))
            history['steps'].append(episode_steps)

        self.env.close()
        
        if log_wandb:
            wandb.finish()

        self.save(f'{self.fname}.h5')

        return history
    
    def save(self, fname):
        if not os.path.exists('./assets'):
            os.mkdir('./assets')

        self.qnet.save(f'./assets/{fname}')

    def load(self, fname):
        self.qnet = load_model(f'./assets/{fname}')