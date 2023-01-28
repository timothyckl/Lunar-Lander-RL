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


    def update_target(self):
        self.target_ddsn.set_weights(self.ddsn.get_weights())