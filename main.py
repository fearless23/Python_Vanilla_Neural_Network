import numpy as np
from data import x, y, test_x, test_y
from network import NeuralNetwork

rate = -0.01
hidden_layers = [2, 4, 3]
layers_afn = ["relu", "relu", "relu","sig"]
epochs = 1200
batch_size = 500

nn = NeuralNetwork(x, y, hidden_layers, layers_afn, rate, "bce")
nn.train(epochs, batch_size, show_Plot=True, showErrors=True)
nn.test(test_x, test_y, True)
