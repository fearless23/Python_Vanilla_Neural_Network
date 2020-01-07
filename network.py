import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, inputs, outputs, layers,
                 learning_rate, loss_fn):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.__basic()
        self._initWeightsAndBiases()

    def __basic(self):
        # Calc some cool things
        self.totalRecords, self.xCols = self.inputs.shape
        N, self.yCols = self.outputs.shape
        if self.layers[0]["dim"] != self.xCols:
            raise Exception('Input dim not equal to input columns.')
        if self.layers[-1]["dim"] != self.yCols:
            raise Exception('Output dim not equal to output columns.')
        if N != self.totalRecords:
            raise Exception('Total Records in input and output are not same')

    def _initWeightsAndBiases(self):
        self.weights = []
        self.biases = []

        for i in range(1, len(self.layers)):
            i_dim, o_dim = self.layers[i-1]["dim"], self.layers[i]["dim"]
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

        # Total cost of batch
        j = np.sum(loss) / batch_size
        dl_dyp = dl_dyp / batch_size
        assert(isinstance(j, float))
        return dl_dyp, j

    def __forwardPass(self, x):
        w, b = self.weights, self.biases
        a, z = [x], []

        for i in range(1, len(self.layers)):
            zi = a[i-1]@w[i-1] + b[i-1]
            ai = self.__actFn(zi, self.layers[i]["act"])
            z.append(zi)
            a.append(ai)

        return a, z

    def __backwardPass(self, a, z, dl_dyp):
        r = self.learning_rate
        w, b = self.weights, self.biases
        dw, db = [], []
        d = dl_dyp
        for k in range(1, len(self.layers)):
            fdashz = self.__actFn_der(z[-k], self.layers[-k]["act"])
            if k == 1:
                d = d * fdashz
            else:
                d = (d@w[1-k].T) * fdashz

            # w[-k] -= r * a[-k-1].T@d
            # b[-k] -= r * np.sum(d, axis=0)

            dwi = -r * a[-k-1].T@d
            dbi = -r * np.sum(d, axis=0)
            dw.insert(0, dwi)
            db.insert(0, dbi)

        # Update Weights
        for i in range(1, len(self.layers)):
            w[-i] = w[-i] + dw[-i]
        # Weights and biases are changed globally at Class
        self.weights = w
        self.biases = b
        return

    def __singleBatchPass(self, x, y, batch_size):
        a, z = self.__forwardPass(x)
        dl_dyp, error = self.__lossCalc(a[-1], y, batch_size)
        self.__backwardPass(a, z, dl_dyp)
        return error

    def __singleEpoch(self, batch_size, total_batches):
        start = 0
        epoch_error = 0.0
        for i in range(total_batches):
            end = int(start + batch_size)
            batchX = self.inputs[start:end, :]
            batchY = self.outputs[start:end, :]
            epoch_error += self.__singleBatchPass(batchX, batchY, batch_size)
            start = int(start + batch_size)

        return epoch_error/total_batches

    def __showPlot(self, xAxisData, yAxisData):
        plt.figure(figsize=(15, 5))
        plt.plot(xAxisData, yAxisData)
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Error for Epoch')
        plt.show()

    def train(self, epochs=1000, batch_size=50,
              show_errors=False, show_plot=True):
        """Train Data over 'n' epochs & 'm' batch_size"""
        N, m = self.totalRecords, batch_size
        if N % m != 0:
            msg = f"Batch Size {m} is not multiple of Total Records({N})."
            raise Exception(msg)
        total_batches = int(N / m)

        epoch_list = []
        error_list = []
        for i in range(1, epochs+1):
            epochAvgError = self.__singleEpoch(m, total_batches)
            epoch_list.append(i)
            error_list.append(epochAvgError)
            if show_errors and (i % 10) == 1:
                print(f"Epoch: {i} Error: {epochAvgError}")

        print(f"Final Error: {error_list[-1]}")
        if show_plot:
            self.__showPlot(epoch_list, error_list)

    def test(self, tx, ty, showPredicted=True,
             metrics=[]):
        test_size, _ = tx.shape
        act = tx
        for i in range(1, len(self.layers)):
            zi = (act@self.weights[i-1]) + self.biases[i-1]
            act = self.__actFn(zi, self.layers[i-1]["act"])

        if showPredicted:
            for i in range(5):
                txi = tx[i:i+1, :]
                tyi = ty[i:i+1, :]
                typi = act[i:i+1, :]
                print(f"#{i+1}: {typi[0]} E:{tyi[0]}\n")

        for metric in metrics:
            if metric == "error":
                # Calculate Accuracy
                _, error = self.__lossCalc(act, ty, test_size)
                print(f"Test Error: {error}")
            if metric == "accuracy":
                print(1)
