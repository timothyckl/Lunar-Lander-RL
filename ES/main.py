import os
import gymnasium as gym

from evolution_strategy.es import EvolutionStrategy
from evolution_strategy.fitness import Fitness
from evolution_strategy.crossover import Crossover
from evolution_strategy.mutation import Mutation
from evolution_strategy.selection import Selection
from evolution_strategy.utils import flatten_weights

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import warnings
warnings.filterwarnings('ignore')


def generate_initial_population(population_size, observation_space, action_space):
    models = []
    initial_population = []

    for _ in range(population_size):
        input_layer = Input(shape=(observation_space,))
        fc1 = Dense(256, activation='relu')(input_layer)
        fc2 = Dense(256, activation='relu')(fc1)
        fc3 = Dense(256, activation='relu')(fc2)
        output_layer = Dense(action_space, activation='linear')(fc3)

        model = Model(inputs=input_layer, outputs=output_layer, trainable=False)
        models.append(model)
        initial_population.append(flatten_weights(models[-1]))

    return initial_population, models[-1]


env = gym.make('LunarLander-v2', render_mode='rgb_array')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

num_generations = 50
population_size = 50  # models per generation
initial_population, model = generate_initial_population(population_size, observation_space, action_space)

fitness_fn = Fitness(model, env)
mutation_fn = Mutation(type='gaussian', mutation_rate=0.1)
crossover_fn = Crossover(crossover_rate=0.5)
selection_fn = Selection(selection_type='rank')

log_freq = 1

es = EvolutionStrategy(
    initial_population, 
    fitness_fn, 
    mutation_fn, 
    crossover_fn, 
    selection_fn, 
    parents_to_keep=5
)

if __name__ == '__main__':
    best_solution, best_fitness = es.run(num_generations, log_frequency=log_freq)
    print(f'\nBest solution: {best_solution}')
    print(f'Best fitness: {best_fitness}\n')