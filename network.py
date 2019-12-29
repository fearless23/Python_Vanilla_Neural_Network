import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, inputs, outputs, hidden_layers,
                 layers_afn, learning_rate, loss_fn):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        # len hiddenlayers+1 = len of layers_afn
        self.layers_afn = layers_afn
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
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
            i_dim, o_dim = self.layers[i-1], self.layers[i]
            p = np.sqrt(1/(i_dim+o_dim))
            self.weights.append(np.random.randn(i_dim, o_dim)*p)
            self.biases.append(np.zeros(shape=(1, o_dim), dtype="float"))

    def __actFn(self, m, act_type):
        mat = m
        if act_type == "relu":
            return np.maximum(0, mat)
        elif act_type == "tanh":
            return (1 - np.exp(-2.0 * mat)) / (1 + np.exp(-2.0*mat))
        else:
            return 1.0 / (1 + np.exp(-1*mat))

    def __actFn_der(self, m, act_type):
        mat = m
        if act_type == "relu":
            return np.where(mat <= 0, 0, 1)
        elif act_type == "tanh":
            return (4 * np.exp(-2.0 * mat)) / ((1 + np.exp(-2.0*mat)**2))
        else:
            return mat * (1 - mat)

    def __lossCalc(self, yp, y, batch_size):
        e = yp - y
        loss, dl_dyp = None, None

        if self.loss_fn == "bce":
            loss = -1*(y * np.log(yp) + (1-y)*np.log(1-yp))
            dl_dyp = -1*(np.divide(y, yp) - np.divide(1 - y, 1 - yp))

        # Default MSE
        else:
            loss = 0.5 * (e**2)
            dl_dyp = e

        error = np.sum(loss)/batch_size
        return dl_dyp, error

    def __forwardPass(self, x):
        w, b = self.weights, self.biases
        a, z = [x], []

        for i in range(1, len(self.layers)):
            zi = a[i-1]@w[i-1] + b[i-1]
            ai = self.__actFn(zi, self.layers_afn[i-1])
            z.append(zi)
            a.append(ai)

        return a, z

    def __backwardPass(self, a, z, dl_dyp, batch_size):
        mul = self.learning_rate / batch_size
        w, b = self.weights, self.biases

        d = dl_dyp
        for k in range(1, len(self.layers)):
            fdashz = self.__actFn_der(z[-k], self.layers_afn[-k])
            if k == 1:
                d = d * fdashz
            else:
                d = (d@w[1-k].T) * fdashz

            w[-k] += mul * a[-k-1].T@d
            b[-k] += mul * np.sum(d, axis=0)

        # Weights and biases are changed globally at Class
        self.weights = w
        self.biases = b
        return

    def __singleBatchPass(self, x, y, batch_size):
        a, z = self.__forwardPass(x)
        dl_dyp, error = self.__lossCalc(a[-1], y, batch_size)
        self.__backwardPass(a, z, dl_dyp, batch_size)
        return error

    def __singleEpoch(self, batch_size, total_batches):
        start = 0
        epoch_error = 0.0
        for i in range(total_batches):
            end = int(start + batch_size)
            batchX = self.inputs[start:end, 0:self.xCols]
            batchY = self.outputs[start:end, 0:self.yCols]
            epoch_error += self.__singleBatchPass(batchX, batchY, batch_size)
            start = int(start + batch_size)

        return epoch_error/total_batches

    def __showPlot(self, xAxisData, yAxisData):
        print(f"Final Error: {yAxisData[-1]}")
        plt.figure(figsize=(15, 5))
        plt.plot(xAxisData, yAxisData)
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Error for Epoch')
        plt.show()

    def train(self, epochs=1000, batch_size=50, showErrors=False,
              show_Plot=True):
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
            if showErrors and ((i+1) % 50) == 1:
                print(f"Epoch: {i+1} Error: {epochAvgError}")
        if show_Plot:
            self.__showPlot(epoch_idx_list, epoch_avg_error_list)

    def test(self, testInput, tesOutput, showPredicted=False):
        act = testInput
        for i in range(1, len(self.layers)):
            zi = (act@self.weights[i-1]) + self.biases[i-1]
            act = self.__actFn(zi, self.layers_afn[i-1])

        batch_size, _ = testInput.shape
        _, error = self.__lossCalc(act, tesOutput, batch_size)
        print(f"Test Error: {error}")
        if showPredicted:
            print(f"Test Output: {act}")
