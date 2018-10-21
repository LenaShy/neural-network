import numpy as np
from image_prep import drawing

img = drawing()
activations = np.random.uniform(0, 0.01, [10000, 1])
weights_first_layer = np.random.uniform(0, 1, [100, 10000])
weights_first_layer = np.random.uniform(0, 1, [100, 10000])
biases = np.random.uniform(0, 5, [100, ])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activations_value(biases, weights, x):
    a = sigmoid(weights.dot(x) + biases)
    return a
