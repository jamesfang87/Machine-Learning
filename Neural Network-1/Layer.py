import numpy as np


class Connection:
    def __init__(self, inputs, outputs, input_size, num_neurons) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.weights = np.random.uniform(-0.5, 0.5, (num_neurons, input_size))
        self.bias = np.zeros((num_neurons, 1))

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, input_error, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error
    