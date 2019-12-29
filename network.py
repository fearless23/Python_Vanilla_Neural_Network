import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, inputs, outputs, hidden_layers,
                 learning_rate, activation="sigmoid"):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.__basic()
        self._initWeightsAndBiases()

    def __basic(self):
        # Calc some cool things
        self.totalRecords, self.xCols = np.shape(self.inputs)
        _, self.yCols = np.shape(self.outputs)
        self.layers = [self.xCols] + self.hidden_layers + [self.yCols]

    def _initWeightsAndBiases(self):
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layers)):
            layer_size = (self.layers[i-1], self.layers[i])
            self.weights.append(np.random.uniform(size=layer_size))
            self.biases.append(np.random.uniform(size=(1, self.layers[i])))

    def __actFn(self, x):
        mat = x
        if self.activation == "reLu":
            return np.maximum(0, mat)
        else:
            return 1.0 / (1 + np.exp(-1*mat))

    def __actFn_der(self, x):
        mat = x
        print(f"MAT: {mat}")
        if self.activation == "reLu":
            # return np.greater(mat, 0).astype(int)
            # return (mat > 0) * 1
            return np.where(mat <= 0, 0, 1)
        else:
            return mat * (1 - mat)

    def __forwardPass(self, x):
        w, b = self.weights, self.biases
        a, z = [x], []

        # Forward Pass
        for i in range(1, len(self.layers)):
            zi = (np.dot(a[i-1], w[i-1])) + b[i-1]
            z.append(zi)
            ai = self.__actFn(zi)
            a.append(ai)

        return a, z

    def __backwardPass(self, a, z, e, batch_size):
        d = []
        mul = self.learning_rate / batch_size
        w, b = self.weights, self.biases
        # Back Propagation
        for k in range(1, len(self.layers)):
            di = None
            fdashz = self.__actFn_der(z[-k])
            if k == 1:
                di = e * fdashz
            else:
                di = (np.dot(d[-1], w[1-k].T)) * fdashz

            d.append(di)
            w[-k] += mul * np.dot(a[-k-1].T, di)
            b[-k] += mul * np.sum(di, axis=0)

        # Weights and biases are changed globally at Class
        self.weights = w
        self.biases = b
        return

    def __singlePass(self, x, y, batch_size):
        a, z = self.__forwardPass(x)
        e = a[-1] - y
        self.__backwardPass(a, z, e, batch_size)
        # Error = Mean Squared Error for this pass
        return 0.5 * np.sum(e**2)

    def __singleEpoch(self, batch_size, total_batches):
        start = 0
        epoch_error = 0.0
        for i in range(total_batches):
            end = int(start + batch_size)
            batchX = self.inputs[start:end, 0:self.xCols]
            batchY = self.outputs[start:end, 0:self.yCols]
            epoch_error += self.__singlePass(batchX, batchY, batch_size)
            start = int(start + batch_size)

        return epoch_error/total_batches

    def __showPlot(self, xAxisData, yAxisData):
        print(f"Final Error: {yAxisData[-1]}")
        plt.figure(figsize=(15, 5))
        plt.plot(xAxisData, yAxisData)
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Error for Epoch')
        plt.show()

    def train(self, epochs=1000, batch_size=50, show_Plot=True):
        total_batches = int(self.totalRecords / batch_size)
        if total_batches < 1:
            print("Batch_Size is not multiple of total Records.")
            return

        epoch_idx_list = []
        epoch_avg_error_list = []
        for i in range(0, epochs):
            epochAvgError = self.__singleEpoch(batch_size, total_batches)
            epoch_idx_list.append(i+1)
            epoch_avg_error_list.append(epochAvgError)
            if ((i+1) % 50) == 1:
                print(f"Epoch: {i+1} Error: {epochAvgError}")
        if show_Plot:
            self.__showPlot(epoch_idx_list, epoch_avg_error_list)

    def test(self, testInput, tesOutput):
        act = testInput
        for i in range(1, len(self.layers)):
            zi = (np.dot(act, self.weights[i-1])) + self.biases[i-1]
            act = self.__actFn(zi)

        e = act - tesOutput
        ssq = np.sum(e**2)
        print(f"Predicted Value: {act}")
        print(f"Prediction Error: {ssq}")
