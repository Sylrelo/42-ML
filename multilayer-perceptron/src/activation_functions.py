import numpy as np


# Plage limitée, introduit une courbure non linéaire
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Valeurs uniquement positives
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


ACTIVATION_FN = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'softmax': softmax
}

ACTIVATION_DERIVATIVE_FN = {
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative,
    'tanh': tanh_derivative,
    'softmax': None
}
