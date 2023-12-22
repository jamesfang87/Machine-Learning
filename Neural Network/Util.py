import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def mse(actual, predicted):
    return np.mean(np.power(actual - predicted, 2))


def mse_prime(actual, predicted):
    return 2 * (predicted - actual) / actual.flatten().size