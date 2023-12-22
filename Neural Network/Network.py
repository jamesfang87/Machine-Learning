import numpy as np
from Layers import FCLayer, ActivationLayer


class Network:
    def __init__(self, epochs, learning_rate, loss_fn, loss_fn_prime):
        self.epochs = epochs
        self.layers = []
        self.lr = learning_rate
        self.L = loss_fn
        self.L_prime = loss_fn_prime

    def forward_propagation(self, data):
        layer_input = data
        for layer in self.layers:
            layer_input = layer.forward_propagation(layer_input)

        output = layer_input
        return output

    def back_propagation(self, network_output, labels):
        error = self.L_prime(labels, network_output)
        for layer in reversed(self.layers):
            if isinstance(layer, FCLayer):
                error = layer.back_propagation(error, self.lr)
            elif isinstance(layer, ActivationLayer):
                error = layer.back_propagation(error)

    def fit(self, training_data, training_labels):
        for i in range(self.epochs):
            accuracy = 0
            for j in range(len(training_data)):
                layer_input = training_data[j]
                output = self.forward_propagation(layer_input)

                if np.argmax(output) == np.argmax(training_labels[j]):
                    accuracy += 1

                self.back_propagation(output, training_labels[j])

            print(f"Completed epoch {i}     Accuracy: {accuracy / len(training_data)}")

    def predict(self, testing_data, testing_labels):
        accuracy = 0
        for i in range(len(testing_data)):
            output = self.forward_propagation(testing_data[i])
            if np.argmax(output) == np.argmax(testing_labels[i]):
                accuracy += 1

        print(f"Accuracy: {accuracy / len(testing_data)}")
