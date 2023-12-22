import numpy as np
from Connection import Connection
from Activations import Activations
from tqdm import tqdm


class NeuralNetwork:
    def __init__(self, epochs, data, learning_rate, num_connections):
        self.epochs = epochs
        self.d = data
        self.learning_rate = learning_rate
        self.num_connections = num_connections
        self.connections = []

    def init_params(self):
        self.connections = [Connection(784, 20),
                            Connection(20, 10)]

    def forward_propagation(self, X):
        inputs = [X] + [0.0 for _ in range(self.num_connections)]  # the last element is a placeholder
        outputs = [0.0 for _ in range(self.num_connections)]
        for i in range(self.num_connections):
            pre_activation = self.connections[i].weights @ inputs[i]
            pre_activation += self.connections[i].bias
            outputs[i] = Activations.sigmoid(pre_activation)
            inputs[i + 1] = outputs[i]

        return inputs, outputs

    def backward_propagation(self, inputs, outputs, label):
        output_gradients = [0.0 for _ in range(self.num_connections)]
        input_gradients = [0.0 for _ in range(self.num_connections)]
        for i in range(self.num_connections - 1, -1, -1):
            if i == self.num_connections - 1:
                dE_dX = outputs[i] - label
            else:
                dE_dY = output_gradients[i + 1]
                dE_dX = self.connections[i + 1].weights.T @ dE_dY * Activations.sigmoid_prime(outputs[i])

            output_gradients[i] = dE_dX
            input_gradients[i] = dE_dX

            # change weights
            weights_gradient = input_gradients[i] @ inputs[i].T
            bias_gradient = input_gradients[i]
            self.connections[i].weights -= self.learning_rate * weights_gradient
            self.connections[i].bias -= self.learning_rate * bias_gradient

    def fit(self):
        for _ in tqdm(range(self.epochs)):
            for X, y in zip(self.d.X_train, self.d.y_train):  # len of training data
                X, y = self.d.convert_to_matrix(X, y)
                inputs, outputs = self.forward_propagation(X)
                self.backward_propagation(inputs, outputs, y)

    def predict(self):
        y_pred = []
        for X, y in zip(self.d.X_test, self.d.y_test):
            X, y = self.d.convert_to_matrix(X, y)
            inputs, outputs = self.forward_propagation(X)
            y_pred.append(np.argmax(outputs[self.num_connections - 1]))

        # display total error
        print(f'testing accuracy: {self.d.calc_accuracy(y_pred, self.d.y_test)}')
