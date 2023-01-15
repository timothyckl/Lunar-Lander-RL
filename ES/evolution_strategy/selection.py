import numpy as np 

class Selection:
    def __init__(self, selection_type='roulette'):
        self.selection_type = selection_type
        self.valid_types = ['roulette', 'rank']

        if self.selection_type not in self.valid_types:
            raise Exception('Invalid selection type')

    def roulette_selection(self, fitness):
        '''
        Roulette wheel selection.
        '''
        fitness_sum = np.sum(fitness)
        length = len(fitness)
        parents = np.zeros(length)

        for i in range(length):
            parents[i] = fitness[i] / fitness_sum

        return parents

    def rank_selection(self, fitness):
        '''
        Rank selection.
        '''
        length = len(fitness)
        parents = np.zeros(length)
        sorted_fitness_indices = np.argsort(fitness)

        for i in range(length):
            parents[sorted_fitness_indices[i]] += 2 * (i + 1) / (length * (length + 1))

        return parents

    def __call__(self, fitness):
        if self.selection_type == 'roulette':
            return self.roulette_selection(fitness)
        elif self.selection_type == 'rank':
            return self.rank_selection(fitness)
        else:
            raise Exception('Invalid selection type')