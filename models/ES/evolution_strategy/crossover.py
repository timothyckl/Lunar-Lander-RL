import numpy as np


class Crossover:
    def __init__(self, crossover_rate):
        self.crossover_rate = crossover_rate

    def crossover(self, parent1, parent2):
        '''
        Crossover two parents to produce a child.
        '''
        if np.random.random() < self.crossover_rate:
            child = np.zeros_like(parent1)
            for i in range(len(parent1)):
                if np.random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
        else:
            child = parent1

        return child

    def __call__(self, parent1, parent2):
        return self.crossover(parent1, parent2)