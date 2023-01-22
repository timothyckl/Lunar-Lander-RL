import os
import numpy as np
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw


class EpisodeSaver:
    def __init__(self, env, frames, algo, episode_number):
        self.env = env
        self.frames = frames
        self.dir = f'./gifs/{algo}/'
        self.episode_number = episode_number
        self.fname = f'episode_{self.episode_number}.gif'

        if not os.path.exists('./gifs'):
            os.mkdir('./gifs')

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

    def label_frames(self):
        labeled_frames = []

        for frame in self.frames:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            # draw on each frame
            draw.text((10, 10), f'Episode: {self.episode_number}', fill=(255, 255, 255))
            labeled_frames.append(np.array(img))

        return labeled_frames

    def save(self):
        labeled_frames = self.label_frames()
        imageio.mimsave(self.dir + self.fname, labeled_frames, fps=60)
        

class ReplayBuffer:
    def __init__(self, max_length, state_size, action_size):
        self.memory_counter = 0
        self.max_length = max_length
        self.state_memory = np.zeros((self.max_length, state_size))
        self.new_state_memory = np.zeros((self.max_length, state_size))
        self.action_memory = np.zeros((self.max_length, action_size), dtype=np.int8)
        self.reward_memory = np.zeros(self.max_length)
        self.done_memory = np.zeros(self.max_length, dtype=np.float32)

    def append(self, state, action, reward, new_state, done):
        idx = self.memory_counter % self.max_length

        self.state_memory[idx] = state
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[idx] = actions
        self.new_state_memory[idx] = new_state
        self.reward_memory[idx] = reward
        self.done_memory[idx] = 1 - done
        self.memory_counter += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_counter, self.max_length)
        sampled_batch = np.random.choice(max_memory, batch_size)
        
        states= self.state_memory[sampled_batch]
        actions = self.action_memory[sampled_batch]
        rewards= self.reward_memory[sampled_batch]
        new_states = self.new_state_memory[sampled_batch]
        dones = self.done_memory[sampled_batch]

        return states, actions, rewards, new_states, dones 