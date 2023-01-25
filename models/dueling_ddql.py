import os
import wandb
import numpy as np
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from .utils import ReplayBuffer, EpisodeSaver
from .dueling_dql import DuelingDQN


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


class DuelingDDQL:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay=1e-3, epsilon_min=0.01, 
                 batch_size=64, memory_size=10_000, update_interval=100):
        self.env = env
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.state_size = self.env.observation_space.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.update_counter = 0
        self.update_interval = update_interval
        self.memory = Buffer(self.memory_size, self.state_size, self.action_size)
        self.qnet = DuelingDQN(self.action_size)
        self.qnet_target = DuelingDQN(self.action_size)

        self.qnet.compile(optimizer=Adam(learning_rate=alpha), loss='mse')

    def remember(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)

    def act(self, state):
        state = np.array([state])
        rand = np.random.random()

        if rand < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            actions = self.qnet.advantage(state)
            return tf.math.argmax(actions, axis=1).numpy()[0]
        
    def update_target(self):
        self.qnet_target.set_weights(self.qnet.get_weights())

    def update(self):
        if self.memory.memory_counter < self.batch_size:
            return
        
        if self.update_counter % self.update_interval == 0:
            self.update_target()

        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        q_eval = self.qnet(states)
        q_next = self.qnet_target(new_states)
        
        q_target = q_eval.numpy()
        max_actions = tf.math.argmax(self.qnet(new_states), axis=1)

        for i, done, in enumerate(dones):
            q_target[i, actions[i]] = rewards[i] + self.gamma * q_next[i, max_actions[i]] * (1 - int(done))

        self.qnet.train_on_batch(states, q_target)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
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

            if log_wandb:
                wandb.log({
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'epsilon': self.epsilon
                })

            if save_episodes:
                if (episode + 1) % save_interval == 0 or (episode == 0):
                    s = EpisodeSaver(self.env, frames, 'Dueling_DDQL', episode + 1)
                    s.save()

            history['reward'].append(episode_reward)
            history['steps'].append(episode_steps)
            history['avg_reward_100'].append(np.mean(history['reward'][-100:]))

            print(f'[EP {episode + 1}/{n_episodes}] - Reward: {episode_reward:.4f} - Steps: {episode_steps} - Eps: {self.epsilon:.4f} - Time: {time() - start_time:.2f}s')

        self.env.close()

        if log_wandb:
            wandb.finish()

        self.save('dueling_ddql')

        return history

    def save(self, fname):
        if not os.path.exists('./assets'):
            os.mkdir('./assets')

        self.qnet.save(f'assets/{fname}', save_format='tf')

    def load(self, fname):
        self.qnet = load_model(fname)