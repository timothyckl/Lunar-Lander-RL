import os
import wandb
import numpy as np
from time import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver


class DDQL:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=0.99, epsilon_min=0.01, 
                 batch_size=64, update_target_interval=100):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_interval = update_target_interval
        self.memory = ReplayBuffer(10_000, self.state_size, self.action_size)
        self.qnet_local = self.create_qnet('qnet_local')
        self.qnet_target = self.create_qnet('qnet_target')

    def create_qnet(self, name):
        model = Sequential([
            Dense(units=256, activation='relu', input_shape=(self.state_size,)),
            Dense(units=256, activation='relu'),
            Dense(units=self.action_size, activation='linear')
        ], name=name)

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)

    def act(self, state):
        '''
        Epsilon-greedy policy is used to choose action.
        This means that if we choose to exploit, we choose the action with the highest Q-value.
        '''
        state = np.reshape(state, [1, self.state_size])
        rand = np.random.random()

        if rand < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.qnet_local.predict_on_batch(state))
        
    def update_local(self):
        if self.memory.memory_counter < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        action_values = np.array(self.action_space, dtype=np.int8)
        action_values = np.dot(actions, action_values)

        q_current = self.qnet_local.predict(new_states, verbose=0)
        q_future = self.qnet_target.predict(new_states, verbose=0)
        q_target = self.qnet_local.predict(states, verbose=0)
        max_actions = np.argmax(q_current, axis=1)

        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_idx, action_values] = rewards + self.gamma * q_future[batch_idx, max_actions.astype(int)] * dones

        self.qnet_local.fit(states, q_target, verbose=0)

        if self.memory.memory_counter % self.update_target_interval == 0:
            self.update_target()

    def update_target(self):
        self.qnet_target.set_weights(self.qnet_local.get_weights())

    def train(self, n_episodes, max_steps, log_wandb=False,
              update=True, save_episodes=False, save_interval=10):
        history = {'reward': [], 'avg_reward_100': [], 'steps': []}

        for episode in range(n_episodes):
            start_time = time()
            state = self.env.reset()
            state = state[0]
            done = False
            episode_reward = 0
            episode_steps = 0
            frames = []

            for _ in range(max_steps):
                action = self.act(state)
                new_state, reward, done, _, _ = self.env.step(action)
                frames.append(self.env.render())

                if update:
                    self.remember(state, action, reward, new_state, done)
                    self.update_local()

                state = new_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if log_wandb:
                wandb.log({
                    'reward': episode_reward, 
                    'steps': episode_steps,
                    'epsilon': self.epsilon,
                })

            if save_episodes:
                if episode % save_interval == 0:
                    s = EpisodeSaver(self.env, frames, 'DDQL', episode + 1)
                    s.save()

            print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

            history['reward'].append(episode_reward)
            history['avg_reward_100'].append(np.mean(history['reward'][-100:]))
            history['steps'].append(episode_steps)

        self.env.close()

        if log_wandb:
            wandb.finish()

        self.save('ddql.h5')
            
        return history

    def save(self, fname):
        if not os.path.exists('./assets/'):
            os.makedirs('./assets')

        self.qnet_local.save(f'./assets/{fname}')

    def load(self, fname):
        self.qnet_local = load_model(f'./assets/{fname}')

        if self.epsilon <= self.epsilon_min:
            self.update_target()
