import numpy as np


class Baseline:
    '''
    # ---------------------------------------------------------
    # Q-learning (off-policy TD control) for estimating π ≈ π∗
    # ---------------------------------------------------------
    # Repeat (for each episode):
        # Initialize S
        # Repeat (for each step in episode):
            # Choose A from S using policy derived from Q (e.g., ε-greedy) 
            # Take action A, observe R, S'
            # Update Q(S, A) ← Q(S, A) + α[R + γmax_a'Q(S', a) - Q(S, A)]
            # Update S ← S'
        # until S is terminal
    '''
    def __init__(self, env, discount, exploration, exploration_decay, exploration_min=0.01):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.discount = discount
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        # self.