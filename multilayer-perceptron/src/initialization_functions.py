import numpy as np


def he_uniform_initialization(n_in, n_out):
    limit = np.sqrt(6 / n_in)
    return np.random.uniform(-limit, limit, (n_in, n_out))


def xavier_uniform_initialization(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


def lecun_uniform_initialization(n_in, n_out):
    limit = np.sqrt(3 / n_in)
    return np.random.uniform(-limit, limit, (n_in, n_out))


def random_initialization(n_in, n_out):
    return np.random.randn(n_in, n_out) * 0.2


INITIALIZATION_FN = {
    'heUniform': he_uniform_initialization,
    'xavierUniform': xavier_uniform_initialization,
    'lecunUniform': lecun_uniform_initialization,
    'random': random_initialization
}