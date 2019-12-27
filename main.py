import numpy as np
import matplotlib.pyplot as plt

# input data
inputs = np.array([[8, 4],
                   [10, 6],
                   [7, 7],
                   [12, 8],
                   [4, 10]
                   ])
inputs = inputs / 12
# output data
outputs = np.array([[80], [90], [70], [60], [50]])
outputs = outputs/100


class NeuralNetwork:

    def __init__(self, inputs, outputs, layers, rate=0.3):
        self.x = inputs
        self.y = outputs
        self.layers = layers
        self.learning_rate = rate
        self.error_history = []
        self.epoch_list = []
        self.weights = []

        for i in range(1, len(layers)):
            layer_size = (self.layers[i-1], self.layers[i])
            self.weights.append(np.random.uniform(size=layer_size))

    def sigmoid(self, m):
        return 1 / (1 + np.exp(-m))

    def sigmoid_der(self, m):
        return m * (1 - m)

    def singlePass(self, epoch):
        z = []
        a = [self.x]
        for i in range(1, len(self.layers)):
            zi = np.dot(a[i-1], self.weights[i-1])
            ai = self.sigmoid(zi)
            z.append(zi)
            a.append(ai)

        e = a[-1] - self.y
        self.error_history.append(np.sum(e**2))
        self.epoch_list.append(epoch)

        ds = []
        # Back - Propagation
        for j in range(1, len(self.layers)):
            fdashz = self.sigmoid_der(z[-j])
            d = e * fdashz
            if j != 1:
                tw = np.transpose(self.weights[1-j])
                d = np.dot(ds[-1], tw) * fdashz
            ds.append(d)
            delta_w = np.dot(np.transpose(a[-j-1]), d)
            self.weights[-j] = self.weights[-j] + self.learning_rate * delta_w

    def trainFor(self, n=1000):
        for i in range(0, n):
            self.singlePass(i+1)
        self.showPlot()

    def showPlot(self):
        print(f"Final Error: {self.error_history[-1]}")
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch_list, self.error_history)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

    def calc(self, test, expected):
        a = [test]
        for i in range(1, len(self.layers)):
            zi = np.dot(a[i-1], self.weights[i-1])
            ai = self.sigmoid(zi)
            a.append(ai)

        e = a[-1] - expected
        ssq = np.sum(e**2)
        print(f"Test Case Error: {ssq}")


nn = NeuralNetwork(inputs, outputs, [2, 2, 3, 1], 0.05)
nn.trainFor()
