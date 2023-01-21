import numpy as np


class MonteCarlo:
    def __init__(self, env, discount, exploration, exploration_decay, exploration_min=0.01):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.discount = discount
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        # self.