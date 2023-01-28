import os
import wandb
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver
from .dueling_dql import DuelingDQN


class Buffer(ReplayBuffer):
    def __init__(self, max_size, input_shape, n_actions):
        super(Buffer, self).__init__(max_size, input_shape, n_actions, is_sarsa=True)

    def append(self, state, action, reward, new_state, new_action, done):
        index = self.memory_counter % self.max_length
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.new_action_memory[index] = new_action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.memory_counter += 1


class DuelingSARSA:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=0.99, epsilon_min=0.01, batch_size=64, replace=100):
        self.env  = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replace = replace

        self.update_counter = 0
        self.memory = Buffer(10_000, self.state_size, self.action_size)
        self.ddsn = DuelingDQN(self.action_size)  # dueling deep sarsa network
        self.target_ddsn = DuelingDQN(self.action_size)

        self.ddsn.compile(optimizer=Adam(learning_rate=alpha), loss='mse')

    def remember(self, state, action, reward, new_state, new_action, done):
        self.memory.append(state, action, reward, new_state, new_action, done)

    def act(self, state):
        state = np.array([state])
        rand = np.random.random()

        if rand < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            actions = self.ddsn.advantage(state)
            return tf.math.argmax(actions, axis=1).numpy()[0]
        
    def update(self):
        if self.memory.memory_counter < self.batch_size:
            return
        
        if self.update_counter % self.replace == 0:
            self.update_target()

        states, actions, rewards, new_states, new_actions, dones = self.memory.sample(self.batch_size)

        q_values = self.ddsn(states)
        q_next = self.target_ddsn(new_states)
        q_target = q_values.numpy()

        for idx, terminal in enumerate(dones):
            if terminal:
                q_target[idx] = 0
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx, new_actions[idx]]

        self.ddsn.train_on_batch(states, q_target)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.update_counter += 1

    def update_target(self):
        self.target_ddsn.set_weights(self.ddsn.get_weights())

    def train(self, n_episodes, max_steps):
        
        
        for episode in range(n_episodes):
            state = self.env.reset()
            state = state[0]
            action = self.act(state)
            done = False
            episode_reward = 0
            episode_steps = 0

            for _ in range(max_steps):
                new_state, reward, done, _, _ = self.env.step(action)
                new_action = self.act(new_state)

                self.remember(state, action, reward, new_state, new_action, done)  
                self.update()

                state = new_state
                action = new_action
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

                print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

        self.env.close()