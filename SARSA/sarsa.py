import os
import random
import time
import numpy as np
from utils import ReplayBuffer, EpisodeSaver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


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