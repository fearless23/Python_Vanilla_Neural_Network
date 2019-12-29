import numpy as np
from data import x, y, test_x, test_y
from network import NeuralNetwork

rate = 0.01
hidden_layers = [2, 7, 6, 4, 2]
epochs = 1
batch_size = 500
activation = "retLu"

nn = NeuralNetwork(x, y, hidden_layers, rate, activation)
nn.train(epochs, batch_size)
nn.test(test_x, test_y)
