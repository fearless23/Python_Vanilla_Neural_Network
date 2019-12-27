from data import inputs, outputs
from network import NeuralNetwork

learning_rate = 0.05
hidden_layers = [2, 3]
epochs = 1000
batch_size = 3

nn = NeuralNetwork(inputs, outputs, hidden_layers, learning_rate)
nn.train(epochs, batch_size)
