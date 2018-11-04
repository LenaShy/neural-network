import numpy as np
from image_prep import drawing
import os

im = drawing()
activations = np.asarray(im).reshape(-1)
weights_first_layer = np.genfromtxt(os.path.abspath("data/weights_first_layer.csv"), delimiter=',')
biases_first_layer = np.genfromtxt(os.path.abspath("data/biases_first_layer.csv"), delimiter=',')



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activations_value(biases, weights, x):
    a = sigmoid(weights.dot(x) + biases)
    return a


print(activations_value(biases_first_layer, weights_first_layer, activations))
