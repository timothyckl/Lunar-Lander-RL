import os
import wandb
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver


class DuelingDQN(Model):
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.v = Dense(1, activation='linear')  # state value stream
        self.a = Dense(action_size, activation='linear')  # advantage value stream

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        
        # Q = V(s) + (A(s, a) - 1/|A| * sum A(s, a'))
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

        return q
    
    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        
        return a


class DuelingDQL:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=0.001,
                  epsilon_min=0.01, batch_size=64, update_interval=100):
        self.env = env
        self.action_size = env.action_space.n
        self.action_space = np.arange(self.action_size)
        self.state_size = env.observation_space.shape[0]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_interval = update_interval

        self.update_counter = 0
        self.memory = ReplayBuffer(10_000, self.state_size, self.action_size)
        self.qnet = DuelingDQN(self.action_size)
        self.qnet_target = DuelingDQN(self.action_size)

        self.qnet.compile(optimizer=Adam(learning_rate=self.alpha), loss='mse')
        self.qnet_target.compile(optimizer=Adam(learning_rate=self.alpha), loss='mse')

    def remember(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        rand = np.random.random()

        if rand < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            actions = self.qnet.advantage(state)
            return tf.math.argmax(actions, axis=1).numpy()[0]
        
    def update(self):
        if self.memory.memory_counter < self.batch_size:
            return
        
        if self.update_counter % self.update_interval == 0:
            self.qnet_target.set_weights(self.qnet.get_weights())

        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        # remove one hot encoding from actions
        action_values = np.argmax(actions, axis=1)

        q_pred = self.qnet(states)
        q_next = tf.math.reduce_max(self.qnet_target(new_states), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        for idx, done in enumerate(dones):
            if done:
                q_target[idx] = 0.0

            q_target[idx, action_values[idx]] = rewards[idx] + self.gamma * q_next[idx]

        self.qnet.train_on_batch(states, q_target)
        self.update_counter += 1

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
                    self.update()

                state = new_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

            if log_wandb:
                wandb.log({
                    'reward': episode_reward, 
                    'steps': episode_steps,
                    'epsilon': self.epsilon,
                })

            if save_episodes:
                if episode % save_interval == 0:
                    s = EpisodeSaver(self.env, frames, 'Dueling_DQL', episode + 1)
                    s.save()

            print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

            history['reward'].append(episode_reward)
            history['avg_reward_100'].append(np.mean(history['reward'][-100:]))
            history['steps'].append(episode_steps)

        self.env.close()

        if log_wandb:
            wandb.finish()

        self.save('dueling_dql')
            
        return history

    def save(self, fname):
        if not os.path.exists('./assets/'):
            os.makedirs('./assets')

        self.qnet.save(f'./assets/{fname}', save_format='tf')

    def load(self, fname):
        self.qnet = load_model(f'./assets/{fname}.tf')