import numpy as np


def binary_cross_entropy(y_true, y_pred):
    pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred))

    return loss


def compute_accuracy(y_pred, y_true):
    _y_to_label = np.argmax(y_true, axis=1)
    _pred = np.argmax(y_pred, axis=1)
    accuracy = np.mean(_pred == _y_to_label)

    return accuracy
