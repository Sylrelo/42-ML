import numpy as np


def sigmoid(x: np.array):
    return 1.0 / (1.0 + np.exp(-x))


def predict(val_x: np.array, weights: np.array, bias=0):
    return sigmoid(val_x.dot(weights) + bias)


def normalize(value, old_min, old_max, new_min, new_max):
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
