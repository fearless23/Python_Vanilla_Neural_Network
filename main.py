from data import inputs, outputs
from network import NeuralNetwork

learning_rate = .005
hidden_layers = [3, 5]
epochs = 1000
batch_size = 2
activation = "sigmoid"

nn = NeuralNetwork(inputs, outputs, hidden_layers, learning_rate, activation)
nn.train(epochs, batch_size)
