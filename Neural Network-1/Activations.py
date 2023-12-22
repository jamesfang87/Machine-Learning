import numpy as np


class Activations:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(pre_activation):
        return 1 / (1 + np.exp(-pre_activation))

    @staticmethod
    def sigmoid_prime(sigmoid):
        return sigmoid * (1 - sigmoid)
