import numpy as np


class Connection:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.5, 0.5, (output_size, input_size))
        self.bias = np.zeros((output_size, 1))
