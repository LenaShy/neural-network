import numpy as np
import cv2
import os


def image_crop():
    img = cv2.imread(os.getcwd() + "/images/test3.png")  # Read in the image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w]  # Crop the image - note we do this on the original image
    cv2.imwrite(os.getcwd() + "test4.png", rect)  # Save the image


activations = np.random.uniform(0, 0.01, [100, 1])
weights = np.random.uniform(0, 1, [10, 100])
biases = np.random.uniform(0, 5, [10, ])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activations_value(biases, weights, x):
    a = sigmoid(weights.dot(x) + biases)
    return a


#print(activations_value(biases, weights, activations))
