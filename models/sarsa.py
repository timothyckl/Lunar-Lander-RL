import os
import random
import time
import numpy as np
from models.dqn import Agent as DQN
from models.utils import ReplayBuffer, EpisodeSaver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class Agent(DQN):
    def __init__(
        self,
        env,  # environment
        alpha,  # learning rate
        gamma,  # discount factor
        epsilon,  # exploration rate
        epsilon_decay,
        update_interval  # update interval for target network
    ):
        super().__init__(env, alpha, gamma, epsilon, epsilon_decay, update_interval)