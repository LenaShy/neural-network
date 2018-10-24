import numpy as np
import cv2
import os
import csv


def set_rand_biases_and_weights():
    with open(os.path.abspath("data/biases_first_layer.csv"), "w") as csv_file:
        biases = np.random.uniform(0, 5, 10000)
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(biases)
    with open(os.path.abspath("data/biases_second_layer.csv"), "w") as csv_file:
        biases = np.random.uniform(0, 5, 100)
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(biases)
    with open(os.path.abspath("data/weights_first_layer.csv"), "w") as csv_file:
        weights_first_layer = np.random.uniform(0, 1, [100, 10000])
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(weights_first_layer)
    with open(os.path.abspath("data/weights_second_layer.csv"), "w") as csv_file:
        weights_second_layer = np.random.uniform(0, 1, [1, 100])
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(weights_second_layer)


'''def learning():

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activations_value(biases, weights, x):
    a = sigmoid(weights.dot(x) + biases)
    return a'''


def create_csv_for_learning(images_dir_name):
    with open(os.path.abspath("data/learning/learning_data.csv"), "w") as csv_file:
        for filename in os.listdir(os.path.abspath("images/learning/{0}/".format(images_dir_name))):
            if filename.endswith(".png"):
                im = cv2.imread("{0}/{1}".format(os.path.abspath("images/learning/{0}/".format(images_dir_name)), filename), 0)
                im = np.asarray(im).reshape(-1)
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(im)


'''Creating csv files from images/learning/l and images/learning/v dirs'''
#create_csv_for_learning('l')
#create_csv_for_learning('v')

'''Setting random biases and weights'''
set_rand_biases_and_weights()