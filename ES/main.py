import numpy as np
import gymnasium as gym

from evolution_strategy.es import EvolutionStrategy
from evolution_strategy.crossover import Crossover
from evolution_strategy.mutation import Mutation
from evolution_strategy.selection import Selection
from evolution_strategy.utils import flatten_weights, reshape_weights 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')


def generate_initial_population(population_size, observation_space, action_space):
    models = []
    initial_population = []

    for i in range(population_size):
        model = Sequential(name=f'model_{i}')
        
        model.add(Dense(64, activation='relu', input_shape=(observation_space,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_space, activation='linear'))
        
        model.trainable = False
        models.append(model)
        initial_population.append(flatten_weights(models[-1]))
    return initial_population, models[-1]


def compute_fitness(model, env, solution):
    '''
    Compute fitness for a single model.
    '''
    fitness = 0
    model.set_weights(reshape_weights(model, solution))

    for _ in range(1):
        state = env.reset()
        state = state[0].reshape(1, env.observation_space.shape[0])
        done = False

        while not done:
            action = np.argmax(model.predict(state, verbose=0))
            state, reward, done, _, _ = env.step(action)
            state = state.reshape(1, env.observation_space.shape[0])
            fitness += reward

        if done:
            print(f'Fitness: {fitness}, done: {done}, model: {model.name}')

    return fitness


env = gym.make('LunarLander-v2')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
population_size = 10
initial_population, model = generate_initial_population(population_size, observation_space, action_space)

fitness_fn = lambda solution: compute_fitness(model, env, solution)
mutation_fn = Mutation(type='gaussian', mutation_rate=0.1)
crossover_fn = Crossover(crossover_rate=0.5)
selection_fn = Selection(selection_type='roulette')

es = EvolutionStrategy(
    initial_population, 
    fitness_fn, 
    mutation_fn, 
    crossover_fn, 
    selection_fn, 
    parents_to_keep=2
)

if __name__ == '__main__':
    es.run(n_generations=1, log_frequency=1)