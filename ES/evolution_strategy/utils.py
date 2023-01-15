import numpy as np
from evolution_strategy.es import EvolutionStrategy

def flatten_weights(model):
    '''
    Flatten the weights of a keras model.
    '''
    weights = np.empty([])

    for layer in model.get_weights():
        weights = np.append(weights, layer)

    return weights


def reshape_weights(model, params):
    '''
    Reshape flattened weights to the shape of keras model.
    '''
    weights = []
    pointer = 0

    for layer in model.get_weights():
        total = 1
        
        for dim in layer.shape:
            total *= dim

        weights.append(params[pointer:pointer + total].reshape(layer.shape))
        pointer += total

    return weights
