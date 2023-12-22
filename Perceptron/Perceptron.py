import numpy as np
import math


def sigmoid(raw):
    var = (1 / (1 + pow(math.e, -raw)))
    return np.where(var > 0.5, 1, 0)


class Perceptron:
    def __init__(self, learning_rate, epochs, activation_fn, data):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_fn = activation_fn
        self.data = data
        self.weights = np.zeros((self.data.X_train.shape[1]))
        self.bias = 0

    def fit(self):
        for _ in range(self.epochs):
            for X, y in zip(self.data.X_train, self.data.y_train):
                prediction = self.predict(X)
                self.weights += self.learning_rate * (y - prediction) * X
                self.bias += self.learning_rate * (y - prediction)

    def predict(self, X):
        return self.activation_fn(self.weights.T @ X + self.bias)
