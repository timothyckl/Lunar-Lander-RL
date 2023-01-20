import numpy as np
from time import time
from .dqn import DQN
from .utils import EpisodeSaver
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


class SARSA(DQN):
    def __init__(self, env, discount, learning_rate, exploration, exploration_decay):
        super().__init__(env, discount, learning_rate, exploration, exploration_decay)
        self.qnet_local = self.create_qnet(name='qnet_local')
        self.qnet_target = self.create_qnet(name='qnet_target')
        

    def update_local(self):
        state, action, reward, next_state, next_action, done = self.memory.sample_SARSA(self.batch_size)

        next_actions_mask = np.tile(np.arange(self.action_size), (self.batch_size, 1)) == \
            np.tile(next_action.reshape(self.batch_size, 1), (1, self.action_size))

        target = reward + self.discount * np.sum(self.qnet_target.predict_on_batch(next_state) * next_actions_mask, axis=1) * (1 - done)
        target_vector = self.qnet_local.predict_on_batch(state)
        idx = [i for i in range(self.batch_size)]
        target_vector[idx, action] = target

        self.qnet_local.fit(state, target_vector, verbose=0)


    def train(self, n_episodes, update_qnets=True):
        '''
        ---------------------------------------------------------
        Sarsa (on-policy TD control) for estimating Q ≈ q∗
        ---------------------------------------------------------
        Repeat (for each episode):
            Initialize S
            Choose A from S using policy derived from Q (e.g., ε-greedy)
            Repeat (for each step in episode):
                Take action A, observe R, S'
                Choose A' from S' using policy derived from Q (e.g., ε-greedy)
                Update Q(S, A) ← Q(S, A) + α[R + γQ(S', A') - Q(S, A)]
                Update S ← S', A ← A'
            until S is terminal
        '''
        for episode in range(n_episodes):
            start_time = time()
            state = self.env.reset()
            state = state[0].reshape(1, self.state_size)
            action = self.act(state)
            done = False
            episode_reward = 0
            episode_steps = 0
            frames = []
            rewards_list = []
            exploration_rate_list = []
            steps_per_episode_list = []
            # tqdm_e = tqdm(range(self.max_steps), desc='Episode {}/{}'.format(episode, n_episodes), leave=False, unit='step')

            for step in range(self.max_steps):
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.state_size)
                next_action = self.act(next_state)
                frame = self.env.render()

                if update_qnets:
                    self.memory.append((state, action, reward, next_state, next_action, done))

                    if len(self.memory) > self.batch_size:
                        self.update_local()

                    if step % self.target_update_interval == 0:
                        self.update_target()

                state = next_state
                action = next_action
                episode_reward += reward
                episode_steps += 1

                rewards_list.append(episode_reward)
                exploration_rate_list.append(self.exploration)
                steps_per_episode_list.append(episode_steps)
                frames.append(frame)

                if done:
                    break

            # update exploration rate
            self.exploration = max(self.exploration_min, self.exploration * self.exploration_decay)

            print(f'[EP {episode + 1}/{n_episodes}]  Rewards: {episode_reward:.4f} | Steps: {episode_steps:.0f} | Eps: {self.exploration:.4f} | Time: {time() - start_time:.4f}s')

            # save the last episode as a gif every 10 episodes
            if ((episode + 1) % 10 == 0) or (episode == 0):
                saver = EpisodeSaver(self.env, frames, algo='SARSA', episode_number=episode + 1)
                saver.save()

        self.env.close()

        return rewards_list, exploration_rate_list, steps_per_episode_list