import os
import pickle
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import deque, namedtuple
from DQN import Agent


env = gym.make('LunarLander-v2')
lr = 0.001
discount = 0.99
exploration_rate = 1.0
exploration_decay = 0.999
update_interval = 5
num_episodes = 2000
agent = Agent(
    env=env,
    alpha=lr,
    gamma=discount,
    epsilon=exploration_rate,
    epsilon_decay=exploration_decay,
    update_interval=update_interval
)

if __name__ == '__main__':
    agent.train(num_episodes=num_episodes)