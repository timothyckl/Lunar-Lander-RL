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
        
    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        # check buffer size before appending
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()

        self.buffer.append(experience)

    def sample(self, batch_size):
        '''
        Sample a batch of experiences from the replay buffer
        '''
        sampled_experience = random.choices(self.buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*sampled_experience)
        
        return (
            np.squeeze(np.array(states)),
            np.squeeze(np.array(actions)),
            np.squeeze(np.array(rewards)),
            np.squeeze(np.array(next_states)),
            np.squeeze(np.array(dones, dtype=np.bool))
        )
    
    def sample_SARSA(self, batch_size):
        '''
        Sample a batch of experiences from the replay buffer
        '''
        sampled_experience = random.choices(self.buffer, k=batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*sampled_experience)
        
        return (
            np.squeeze(np.array(states)),
            np.squeeze(np.array(actions)),
            np.squeeze(np.array(rewards)),
            np.squeeze(np.array(next_states)),
            np.squeeze(np.array(next_actions)),
            np.squeeze(np.array(dones, dtype=np.bool))
        )