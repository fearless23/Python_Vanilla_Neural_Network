import random
import numpy as np
from numpy import loadtxt


def gen_data(n=10000):
    gx = []
    gy = []
    for i in range(n):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        gx.append([a/100, b/100])
        gy.append([(a+2*b)/300])
    return np.array(gx, dtype="float32"), np.array(gy)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# raw_x, raw_y = gen_data(5000)
# raw_tx, raw_ty = gen_data(600)

# x, y, tx, ty = raw_x, raw_y, raw_tx, raw_ty

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
np.random.shuffle(dataset)
k = 700
raw_x = dataset[:k, 0:8]
raw_tx = dataset[k:, 0:8]
raw_y = dataset[:k, 8:]
raw_ty = dataset[k:, 8:]

x = normalized(raw_x, 0)
y = normalized(raw_y, 1)
tx = normalized(raw_tx, 0)
ty = normalized(raw_ty, 1)
# '''
print("2 Samples of Data")
print(f"raw_x:\n{raw_x[0:2, :]}\n")
print(f"norm_x:\n{x[0:2, :]}\n")
print(f"raw_y:\n{raw_y[0:2, :]}\n")
print(f"norm_y:\n{y[0:2, :]}\n")
# '''
