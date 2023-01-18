# redo

import os
import random
import time
import numpy as np
from models.utils import ReplayBuffer, EpisodeSaver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import warnings
warnings.filterwarnings('ignore')


class DQN:
    def __init__(
        self,
        env,  # environment
        alpha,  # learning rate
        gamma,  # discount factor
        epsilon,  # exploration rate
        epsilon_decay, 
        update_interval: int = 100 # update interval for target network
    ):
        self.env = env 
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.update_interval = update_interval

        self.memory = ReplayBuffer()
        self.batch_size = 100
        self.epsilon_min = 0.01
        self.max_steps_per_episode = 10_000

        self.qnet_local = self.create_qnet(name='qnet_local')
        self.qnet_target = self.create_qnet(name='qnet_target')

    def create_qnet(self, name):
        model = Sequential([
            Dense(units=64, activation='relu', input_shape=(self.state_size,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=self.action_size, activation='linear')  # linear activation for Q(s, a)
        ], name=name)

        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.alpha))

        return model
        
    def save_weights(self, fname):

        if not os.path.exists('assets/'):
            os.mkdir('assets/')

        self.qnet_local.save_weights(f'assets/{fname}')

    def act(self, state):
        if random.random() > self.epsilon:
            # exploit
            return np.argmax(self.qnet_local.predict(state, verbose=0))
        else:
            # explore
            return self.env.action_space.sample()

    def update_local(self):
        # sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Q(s, a) = r
        q_values = self.qnet_local.predict(states, verbose=0)
        # Q(s', a')
        q_values_next = self.qnet_target.predict(next_states, verbose=0)
        q_values_next[dones] = 0.0
        # Q(s, a) = r + gamma * max(Q(s', a'))
        q_values[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(q_values_next, axis=1)

        # train the local network
        self.qnet_local.fit(states, q_values, verbose=0)

    def update_target(self):
        self.qnet_target.set_weights(self.qnet_local.get_weights())

    def train(self, num_episodes):
        rewards_list = []
        exploration_rate_list = []
        steps_per_episode_list = []

        for episode in range(num_episodes):
            start_time = time.time()
            state, _ = self.env.reset()
            state = state.reshape(1, self.state_size)
            episode_reward = 0
            episode_steps = 0
            frames = []

            for step in range(self.max_steps_per_episode):
                action = self.act(state)

                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.state_size)
                
                frames.append(self.env.render())

                # store experience in replay buffer
                self.memory.append((state, action, reward, next_state, done))

                episode_reward += reward
                episode_steps += 1
                state = next_state

                self.update_local()

                # Every k steps, copy actual network weights to the target network weights
                if step % self.update_interval == 0:
                    self.update_target()

                if done:
                    break

            # decay the epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            rewards_list.append(episode_reward)
            exploration_rate_list.append(self.epsilon)
            steps_per_episode_list.append(episode_steps)

            # check for terminal condition
            avg_reward = np.mean(rewards_list[-100:])
            # if avg_reward >= 200.0:
            #     print(f'Environment solved in {episode + 1} episodes with avg reward {avg_reward}')
            #     self.save_weights(f'qnet_ep_{episode}.h5')
            #     break

            print(f'\n[Episode {episode + 1}/{num_episodes}]\nReward: {episode_reward:.4f}   Avg Reward: {avg_reward:.4f}    Steps: {episode_steps:.0f}    ER: {self.epsilon:.4f}    Time: {(time.time() - start_time):.4f}s')

            # save the last episode as a gif every 10 episodes
            if ((episode + 1) % 10 == 0) or (episode == 0):
                saver = EpisodeSaver(self.env, frames, algo='DQN', episode_number=episode + 1)
                saver.save()

        self.env.close()
        print('Training complete...')
        self.save_weights(f'qnet_local_ep_{episode}.h5')

        return rewards_list, exploration_rate_list, steps_per_episode_list