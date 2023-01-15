import os
import numpy as np
from time import time
from tqdm import tqdm


class EvolutionStrategy:
    def __init__(
        self,
        initial_population,
        fitness_function,
        mutation_function,
        crossover_function,
        selection_function,
        parents_to_keep=0,
        callbacks=None 
    ):
        self.population = initial_population
        self.fitness_fn = fitness_function
        self.mutation_fn = mutation_function
        self.crossover_fn = crossover_function
        self.selection_fn = selection_function
        self.parents_to_keep = parents_to_keep
        self.callbacks = callbacks
        
        self.n_generations = 0
        self.fitness_history = []
        self.fitness = None
        self.best_fitness = None
        self.best_solution = None
        
    def compute_fitness(self):
        '''
        Compute fitness for each solution in the population.
        '''
        self.fitness = list(map(self.fitness_fn, self.population))  
        current_best_fitness = np.max(self.fitness)
        print(f'Current best fitness: {current_best_fitness}')

        if self.best_fitness is None or current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = self.population[np.argmax(self.fitness)]

        self.fitness_history.append(
            (
                self.n_generations, 
                current_best_fitness, 
                np.mean(self.fitness), 
                np.std(self.fitness)
            )
        )

    def generate_new_population(self):
        '''
        Generate new population of solutions.
        '''
        population_size = len(self.population)
        parent_probability = self.selection_fn(self.fitness)
        
        elites = []
        elite_indices = np.argsort(parent_probability)[:self.parents_to_keep]

        for idx in elite_indices:
            elites.append(self.population[idx])

        children = []

        for _ in range(population_size - self.parents_to_keep):
            parent_indices = np.random.choice(
                population_size,
                size=2,
                replace=False,
                p=parent_probability
            )
            parent1, parent2 = self.population[parent_indices[0]], self.population[parent_indices[1]]
            child = self.crossover_fn(parent1, parent2)
            child = self.mutation_fn(child)
            children.append(child)

        self.population = elites + children

    def run(self, n_generations, log_frequency=10):
        '''
        Run the evolution strategy for n_generations.
        '''
        
        self.compute_fitness()  # compute fitness for initial population
        
        start_time = time()

        for _ in tqdm(range(n_generations)):
            self.generate_new_population()
            self.compute_fitness()
            self.n_generations += 1

            # if self.num_generations % log_frequency == 0:
            print(f'[Generation {self.n_generations}]  {time() - start_time:.2f}s')
            print(f'Best fitness: {self.best_fitness:.2f}')
            print(f'Average fitness: {np.mean(self.fitness):.2f}')
            print(f'Std. fitness: {np.std(self.fitness):.2f}')

