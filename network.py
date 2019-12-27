import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, inputs, outputs, hidden_layers, learning_rate=0.3):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.__basic()

    def __basic(self):
        # Calc some cool things
        totalRecords, xCols = np.shape(self.inputs)
        _, yCols = np.shape(self.outputs)
        self.totalRecords = totalRecords
        self.xCols = xCols
        self.yCols = yCols
        self.layers = [xCols]+self.hidden_layers+[yCols]
        self.error_history = []
        self.epoch_list = []
        self.weights = []
        self.biases = []

        for i in range(1, len(self.layers)):
            layer_size = (self.layers[i-1], self.layers[i])
            self.weights.append(np.random.uniform(size=layer_size))
            self.biases.append(np.random.uniform(size=(1, self.layers[i])))

    def __sigmoid(self, m):
        return 1 / (1 + np.exp(-m))

    def __sigmoid_der(self, m):
        return m * (1 - m)

    def __singlePass(self):
        # Stored weighted inputs, activated and deltas
        z = []
        a = [self.x]
        ds = []

        # Move Forward
        for i in range(1, len(self.layers)):
            zi = (np.dot(a[i-1], self.weights[i-1])) + self.biases[i-1]
            ai = self.__sigmoid(zi)
            z.append(zi)
            a.append(ai)

        # Calc error Vector
        e = a[-1] - self.y

        # Back Propagation
        for j in range(1, len(self.layers)):
            fdashz = self.__sigmoid_der(z[-j])
            d = e * fdashz
            if j != 1:
                tw = np.transpose(self.weights[1-j])
                d = np.dot(ds[-1], tw) * fdashz

            ds.append(d)
            delta_w = np.dot(np.transpose(a[-j-1]), d)
            self.weights[-j] = self.weights[-j] + self.learning_rate * delta_w

            delta_b = np.sum(delta_w, axis=0)
            self.biases[-j] = self.biases[-j] + self.learning_rate * delta_b

        # All Weights and biases are changed globally...
        # Return Error in this pass
        return np.sum(e**2)

    def train(self, epochs=1000, batch_size=50):
        for i in range(0, epochs):
            avgError = self.__trainInBatches(batch_size=batch_size)
            self.error_history.append(avgError)
            self.epoch_list.append(i+1)
        self.__showPlot()

    def __trainInBatches(self, batch_size=50):
        no_of_batches = int(self.totalRecords / batch_size)
        start = 0
        error = 0
        for i in range(no_of_batches):
            end = start + batch_size
            self.x = self.inputs[start:end, 0:self.xCols]
            self.y = self.outputs[start:end, 0:self.yCols]
            error += self.__singlePass()
            start = start + batch_size

        return error/no_of_batches

    def __showPlot(self):
        print(f"Final Error: {self.error_history[-1]}")
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch_list, self.error_history)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

    def test(self, testInput, tesOutput):
        a = [testInput]
        for i in range(1, len(self.layers)):
            zi = np.dot(a[i-1], self.weights[i-1])
            ai = self.__sigmoid(zi)
            a.append(ai)

        e = a[-1] - tesOutput
        ssq = np.sum(e**2)
        print(f"Test Case Error: {ssq}")
