import numpy as np
from data import x, y, test_x, test_y
from network import NeuralNetwork

rate = 0.1
layers = [
    {"dim": 2, "act": None},
    {"dim": 2, "act": "tanh"},
    # {"dim": 4, "act": "relu"},
    {"dim": 2, "act": "tanh"},
    {"dim": 1, "act": "sig"}
]
epochs = 1200
batch_size = 500
loss_fn = "bce"
nn = NeuralNetwork(x, y, layers, rate, loss_fn)
nn.train(epochs, batch_size, show_Plot=True, showErrors=True)
nn.test(test_x, test_y, True)
