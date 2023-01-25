import os
from time import time
import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver

class DuelingDeepQNetwork(Model):
    def __init__(self, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.V = Dense(1, activation='linear')
        self.A = Dense(n_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A


class Buffer(ReplayBuffer):
    def __init__(self, max_size, input_shape, n_actions):
        super(Buffer, self).__init__(max_size, input_shape, n_actions)

    def append(self, state, action, reward, state_, done):
        index = self.memory_counter % self.max_length
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.memory_counter += 1


class DuelingDQL:
    def __init__(self, env, alpha, gamma, epsilon, batch_size=64,
                epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=10_000, replace=100):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.update_counter = 0
        self.memory = Buffer(mem_size, self.state_size, self.action_size)
        self.q_eval = DuelingDeepQNetwork(self.action_size)
        self.q_next = DuelingDeepQNetwork(self.action_size)

        self.q_eval.compile(optimizer=Adam(learning_rate=alpha), loss='mse')

    def remember(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)

    def act(self, observation):
        state = np.array([observation])
        rand = np.random.random()
        
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def update(self):
        if self.memory.memory_counter < self.batch_size:
            return

        if self.update_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.update_counter += 1

    def train(self, n_episodes, max_steps, log_wandb=False, 
              update=True, save_episodes=False, save_interval=100):
        history = {'reward': [], 'avg_reward_100': [], 'steps': []}

        for episode in range(n_episodes):
            start_time = time()
            state = self.env.reset()
            state = state[0]
            done = False
            episode_reward = 0
            episode_steps = 0
            frames= []

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

            if log_wandb:
                wandb.log({
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'epsilon': self.epsilon
                })

            if save_episodes:
                if (episode + 1) % save_interval == 0 or (episode == 0):
                    s = EpisodeSaver(self.env, frames, 'DuelingDQL', episode + 1)
                    s.save()

            history['reward'].append(episode_reward)
            history['avg_reward_100'].append(np.mean(history['reward'][-100:]))
            history['steps'].append(episode_steps)
            
            print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

        self.env.close()

        if log_wandb:
            wandb.finish()

        self.save_model('dueling_dqn')

        return history

    def save_model(self, fname):
        if not os.path.exists('./assets'):
            os.mkdir('./assets')

        self.q_eval.save(f'assets/{fname}', save_format='tf')

    def load_model(self, fname):
        self.q_eval = load_model(fname)