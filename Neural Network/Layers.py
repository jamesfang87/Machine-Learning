import numpy as np


class FCLayer:
    def __init__(self, input_size, num_neurons):
        self.inputs = None
        self.input_size = input_size
        self.output_size = num_neurons

        self.weights = np.random.rand(input_size, num_neurons) - 0.5
        self.bias = np.random.rand(1, num_neurons) - 0.5

    def forward_propagation(self, inputs):
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def back_propagation(self, output_derivative, lr):
        # calculate gradients
        input_derivative = output_derivative @ self.weights.T  # dE/dx, used for the next layer's dE/dY
        weight_gradient = self.inputs.T @ output_derivative  # dE/dW_i, used in gradient descent to update weights
        bias_gradient = output_derivative  # dE/dB_i, used in the gradient descent to update bias

        # change weights
        self.weights -= lr * weight_gradient
        self.bias -= lr * bias_gradient

        # pass dE/dX to the next layer
        return input_derivative


class ActivationLayer:
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

        self.input = None

    def forward_propagation(self, inputs):
        self.input = inputs
        return self.f(inputs)

    def back_propagation(self, output_derivative):
        return self.f_prime(self.input) * output_derivative