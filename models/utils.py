import os
import random
import numpy as np
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
from collections import deque, namedtuple
from .sumTree import SumTree


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

class PrioritizedBuffer:
    def __init__(self, max_length=1_000, eps=1e-2, alpha=0.1, beta=0.1):
        self.buffer = SumTree(size=max_length)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        self.count = 0
        self.real_size = 0
        self.max_length = max_length

    def append(self, experience):
        '''
        Append a batch of experiences to the replay buffer
        '''
        state, action, reward, next_state, done = experience

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in buffer
        self.buffer[self.count] = (state, action, reward, next_state, done)

        # update count
        self.count += 1
        self.real_size = min(self.real_size + 1, self.max_length)

    def sample(self, batch_size):
        '''
        Sample a batch of experiences from the replay buffer
        '''
        # calculate priority segment
        priority_segment = self.tree.total() / batch_size

        # calculate importance sampling weights
        weights = []
        idxs = []
        p_min = self.tree.min() / self.tree.total()
        max_weight = (p_min * self.real_size) ** (-self.beta)

        for i in range(batch_size):
            # sample a random priority
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = random.uniform(a, b)

            # retrieve transition index and priority
            idx, priority, data = self.tree.get(value)
            idxs.append(idx)

            # calculate importance sampling weight
            p_sample = priority / self.tree.total()
            weight = (p_sample * self.real_size) ** (-self.beta)
            weights.append(weight / max_weight)

        # convert to numpy arrays
        weights = np.array(weights, dtype=np.float32)
        idxs = np.array(idxs, dtype=np.int32)

        # retrieve sampled experiences with corresponding weights
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in idxs])

        return (
            np.squeeze(np.array(states)),
            np.squeeze(np.array(actions)),
            np.squeeze(np.array(rewards)),
            np.squeeze(np.array(next_states)),
            np.squeeze(np.array(dones, dtype=np.bool)),
        ), weights, idxs
        

        