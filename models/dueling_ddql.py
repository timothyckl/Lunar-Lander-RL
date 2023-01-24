import os
import wandb
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver
from dueling_ddql import DuelingDQN


class DuelingDDQL:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=0.99,
                  epsilon_min=0.01, batch_size=64, update_interval=100):
        self.env = env
        self.action_size = env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.state_size = env.observation_space.shape[0]
        self.alpha = alpha