from Data import Data
from NeuralNetwork import NeuralNetwork


d = Data('troll.csv')
d.get_mnist()
n = NeuralNetwork(10, d, 0.1, 2)
n.init_params()
n.fit()
n.predict()
