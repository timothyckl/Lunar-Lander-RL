import gymnasium as gym

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class ReplayBuffer:
    def __init__(self, max_length=1_000_000):
        self.buffer = deque(maxlen=max_length)
        self.memory = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # sampled_exp contains a namedtuple
        sampled_exp = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sampled_exp)

        return (
            np.squeeze(states),
            np.array(actions),
            np.array(rewards),
            np.squeeze(next_states),
            np.array(dones, dtype=np.bool),
        )


class Agent:
    def __init__(
        self,
        env,  # environment
        alpha,  # learning rate
        gamma,  # discount factor
        epsilon,  # exploration rate
        epsilon_decay, 
        update_interval  # update interval for target network
    ):
        self.env = env 
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.update_interval = update_interval

        self.rewards_list = []
        self.memory = ReplayBuffer()
        self.batch_size = 100
        self.epsilon_min = 0.01
        self.max_steps_per_episode = 10_000

        self.qnet_local = self.create_qnet()
        self.qnet_target = self.create_qnet()

    def create_qnet(self):
        model = Sequential(
            Dense(units=64, activation='relu', input_shape=(self.state_size,)),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=self.action_size, activation='linear')
        , name='Q_Network')

        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.lr))

        return model
        
    def act(self, state):
        if random.random() > self.epsilon:
            # exploit
            return np.argmax(self.qnet_local.predict(state))
        else:
            # explore
            return self.env.action_space.sample()

    def update_local(self):
        states, actions, rewards, next_states, dones= self.buffer.sample(self.batch_size)

        # Q(s, a)
        q_values = self.qnet_local.predict(states)
        # Q(s', a')
        q_values_next = self.qnet_target.predict(next_states)
        q_values_next[dones] = 0.0
        # Q(s, a) = r + gamma * max(Q(s', a'))
        q_values[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(q_values_next, axis=1)

        # train the local network
        self.qnet_local.fit(states, q_values, verbose=0)

    def update_target(self):
        self.qnet_target.set_weights(self.qnet_local.get_weights())

    def train(self, num_episodes):
        for episode in num_episodes:
            state = self.env.reset()
            state = state.reshape(1, self.state_size)
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                action = self.act(state)

                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.state_size)
                
                # store experience in replay buffer
                self.memory.append((state, action, reward, next_state, done))

                episode_reward += reward
                state = next_state

                if len(self.memory) > self.batch_size:
                    self.update_local()

                # Every k steps, copy actual network weights to the target network weights
                if step % self.update_interval == 0:
                    self.update_target()

                if done:
                    break

            self.rewards_list.append(episode_reward)
            # decay the epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # check for terminal condition
            avg_reward = np.mean(self.rewards_list[-100:])
            if avg_reward > 200.0:
                print(f'Environment solved in {episode} episodes with avg reward {avg_reward}')
                break