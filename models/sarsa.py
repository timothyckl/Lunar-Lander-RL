from dqn import DQN


class SARSA(DQN):
    def __init__(self, env, discount, learning_rate, exploration, exploration_decay):
        super().__init__(env, discount, learning_rate, exploration, exploration_decay)

    # def update_local(self, state, action, reward, next_state, next_action, done):

    # ...

    def train(self, n_episodes, update_qnets=True):
        '''
        ---------------------------------------------------------
        Sarsa (on-policy TD control) for estimating Q ≈ q∗
        ---------------------------------------------------------
        '''