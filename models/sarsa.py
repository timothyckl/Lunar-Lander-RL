import os
import random
import time
import numpy as np
from models.dqn import DQN
from models.utils import EpisodeSaver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


# create a class that subclasses the DQN class
class SARSA(DQN):
    def __init__(
        self,
        env,  # environment
        alpha,  # learning rate
        gamma,  # discount factor
        epsilon,  # exploration rate
        epsilon_decay
    ):
        super().__init__(
            env=env,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay
        )

    def update_local(self):
        '''
        Update local network weights with SARSA algorithm
        '''
        # sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, \
            next_actions, dones = self.memory.sarsa_sample(self.batch_size)

        # Q(s, a)
        q_values = self.qnet_local.predict_on_batch(states)
        # Q(s', a')
        q_values_next = self.qnet_local.predict_on_batch(next_states)
        # update Q(s, a) with SARSA algorithm
        q_values[np.arange(self.batch_size), actions] = rewards + \
            self.gamma * q_values_next[np.arange(self.batch_size), next_actions] * (1 - dones)

        self.qnet_local.fit(states, q_values, epochs=1, verbose=0)

    def train(self, num_episodes=1000, max_num_steps=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            state = state[0].reshape(1, self.state_size)
            episode_reward = 0
            frames = []

            for step in range(max_num_steps):
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.state_size)
                next_action = self.act(next_state)

                frames.append(self.env.render())

                self.memory.append((state, action, reward, next_state, next_action, done))

                state = next_state
                episode_reward += reward

                self.update_local()

                if done:
                    break

            # decay the epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            print(f'\n[Episode {episode + 1}/{num_episodes}]\nReward: {episode_reward:.4f}')

            # save the last episode as a gif every 10 episodes
            if ((episode + 1) % 10 == 0) or (episode == 0):
                saver = EpisodeSaver(self.env, frames, algo='DQN', episode_number=episode + 1)
                saver.save()

            self.env.close()
            print('Training complete...')
            self.save_weights(f'qnet_local_ep_{episode}.h5')