import numpy as np
from data import x, y, tx, ty
from network import NeuralNetwork

rate = 0.001
layers = [
    {"dim": 8, "act": None},
    {"dim": 12, "act": "relu"},
    {"dim": 8, "act": "relu"},
    {"dim": 1, "act": "sig"}
]
loss_fn = "bce"
nn = NeuralNetwork(x, y, layers, rate, loss_fn)

epochs = 500
batch_size = 20
nn.train(epochs, batch_size, show_plot=True, show_errors=True)
nn.test(tx, ty, True, ["accuracy"])
