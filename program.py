import numpy as np
from image_prep import drawing
import pandas as pd
import os
import cv2

filepath = drawing()
im = cv2.imread(filepath, 0)
activations = np.asarray(im).reshape(-1)
weights_first_layer = pd.read_csv(os.path.abspath("data/weights_first_layer.csv"))
weights_second_layer = pd.read_csv(os.path.abspath("data/weights_second_layer.csv"))
biases_first_layer = pd.read_csv(os.path.abspath("data/biases_first_layer.csv"))
biases_second_layer = pd.read_csv(os.path.abspath("data/biases_second_layer.csv"))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activations_value(biases, weights, x):
    a = sigmoid(weights.dot(x) + biases)
    return a


first_layer = activations_value(biases_first_layer, weights_first_layer, activations)
print(activations_value(biases_second_layer, weights_second_layer, first_layer))
