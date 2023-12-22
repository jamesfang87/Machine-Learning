from Network import Network
from Layers import *
from Util import *
import pandas as pd
from keras.utils import to_categorical


network = Network(30,  0.5, mse, mse_prime)
network.layers = [
                    FCLayer(28 * 28, 20),
                    ActivationLayer(tanh, tanh_prime),
                    FCLayer(20, 10),
                    ActivationLayer(tanh, tanh_prime),
                  ]

train = pd.read_csv(r"MNIST/mnist_train.csv")
test = pd.read_csv(r"MNIST/mnist_test.csv")


training_data = np.reshape(np.array(train.iloc[:, 1:]), (60000, 1, 28 * 28)) / 255
training_labels = to_categorical(np.array(train.iloc[:, 0]))

testing_data = np.reshape(np.array(test.iloc[:, 1:]), (10000, 1, 28 * 28)) / 255
testing_labels = to_categorical(np.array(test.iloc[:, 0]))


network.fit(testing_data, testing_labels)
network.predict(testing_data, testing_labels)
