import numpy as np
import gymnasium as gym

from evolution_strategy.es import EvolutionStrategy
from evolution_strategy.crossover import Crossover
from evolution_strategy.mutation import Mutation
from evolution_strategy.selection import Selection
from evolution_strategy.utils import flatten_weights, reshape_weights 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import warnings
warnings.filterwarnings('ignore')


def generate_initial_population(population_size, observation_space, action_space):
    models = []
    initial_population = []

    for _ in range(population_size):
        input_layer = Input(shape=(observation_space,))
        fc1 = Dense(64, activation='relu')(input_layer)
        fc2 = Dense(64, activation='relu')(fc1)
        fc3 = Dense(64, activation='relu')(fc2)
        output_layer = Dense(action_space, activation='linear')(fc3)

        model = Model(inputs=input_layer, outputs=output_layer, trainable=False)
        models.append(model)
        initial_population.append(flatten_weights(models[-1]))

    return initial_population, models[-1]


def compute_fitness(model, env, solution):
    '''
    Compute fitness for a single model.
    '''
    fitness = 0
    model.set_weights(reshape_weights(model, solution))
    state_size = env.observation_space.shape[0]

    for _ in range(1):
        state = env.reset()
        state = state[0].reshape(1, state_size)
        done = False

        while not done:
            action = np.argmax(model.predict(state, verbose=0))
            state, reward, done, _, _ = env.step(action)
            state = state.reshape(1, state_size)
            fitness += reward

    return fitness


env = gym.make('LunarLander-v2')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

num_generations = 100
population_size = 10  # models per generation
initial_population, model = generate_initial_population(population_size, observation_space, action_space)

fitness_fn = lambda solution: compute_fitness(model, env, solution)
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