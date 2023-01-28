import os
import wandb
import numpy as np
from time import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver

class SARSA:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=0.99, epsilon_min=0.01, batch_size=64):
        self.env = env 
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.state_size = self.env.observation_space.shape[0]
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = ReplayBuffer(10_000, self.state_size, self.action_size, is_sarsa=True)
        self.dsn = self.create_dsn('dsn')

    def create_dsn(self, name):
        model = Sequential([
            Dense(units=256, activation='relu', input_shape=(self.state_size,)),
            Dense(units=256, activation='relu'),
            Dense(units=self.action_size, activation='linear')
        ], name=name)

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model

    def remember(self, state, action, reward, new_state, done, new_action):
        self.memory.append(state, action, reward, new_state, done, new_action)

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
            return np.argmax(self.dsn.predict(state, verbose=0))

    def update(self):
        if self.memory.memory_counter > self.batch_size:
            state, action, reward, new_state, new_actions, done = self.memory.sample(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_idx = np.dot(action, action_values)

            new_action_values = np.array(self.action_space, dtype=np.int8)
            new_action_idx = np.dot(new_actions, new_action_values)

            q_values = self.dsn.predict(state)
            q_values_next = self.dsn.predict(new_state)

            q_update = reward + self.gamma * q_values_next[np.arange(self.batch_size), new_action_idx] * done
            q_values[np.arange(self.batch_size), action_idx] = q_update

            self.dsn.fit(state, q_values, verbose=0)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def train(self, n_episodes, max_steps=1000,log_wandb=False, 
                update=True, save_episodes=False, save_interval=10):
        history = {'reward': [], 'avg_reward_100': [], 'steps': []}

        for episode in range(n_episodes):
            start_time = time()
            done = False
            episode_reward = 0
            episode_steps = 0
            frames = []

            state = self.env.reset()
            state = state[0]
            action = self.act(state)

            for _ in range(max_steps):
                new_state, reward, done, _, _ = self.env.step(action)
                new_action = self.act(new_state)
                frames.append(self.env.render())
                
                if update:
                    self.update()
                    self.remember(state, action, reward, new_state, done, new_action)

                state = new_state
                action = new_action
                episode_reward += reward
                episode_steps += 1

                if done:
                    break
                    
            if log_wandb:
                wandb.log({
                    'reward': episode_reward,
                    'steps': episode_steps, 
                    'epsilon': self.epsilon
                })
                
            if save_episodes:
                if (episode + 1) % save_interval == 0 or (episode == 0):
                    s = EpisodeSaver(self.env, frames, 'SARSA', episode + 1)
                    s.save()
                
            print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

            history['reward'].append(episode_reward)
            history['avg_reward_100'].append(np.mean(history['reward'][-100:]))
            history['steps'].append(episode_steps)

        self.env.close()
        
        if log_wandb:
            wandb.finish()
            
        self.save('sarsa.h5')
        
        return history

    def save(self, fname):
        if not os.path.exists('./assets'):
            os.mkdir('./assets')

        self.dsn.save(f'./assets/{fname}')

    def load(self, fname):
        self.dsn = load_model(f'./assets/{fname}')