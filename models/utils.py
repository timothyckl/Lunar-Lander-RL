
import os
import random
import numpy as np
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
from collections import deque, namedtuple


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
    def __init__(self, max_length=1_000):
        self.buffer = deque(maxlen=max_length)
        self.memory = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        '''
        Sample a batch of experiences from the replay buffer
        '''
        sampled_exp = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sampled_exp)

        return (
            np.squeeze(states),
            np.array(actions),
            np.array(rewards),
            np.squeeze(next_states),
            np.array(dones, dtype=np.bool)
        )

    def sarsa_sample(self, batch_size):
        '''
        Sample a batch of experiences from the replay buffer for SARSA
        '''

        # overwrite the memory namedtuple
        self.memory = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'next_action', 'done'])
        print(self.buffer)
        sampled_exp = random.sample(self.buffer, batch_size)
        print(sampled_exp)
        states, actions, rewards, next_states, next_actions, dones = zip(*sampled_exp)

        return (
            np.squeeze(states),
            np.array(actions),
            np.array(rewards),
            np.squeeze(next_states),
            np.array(next_actions),
            np.array(dones, dtype=np.bool)
        )