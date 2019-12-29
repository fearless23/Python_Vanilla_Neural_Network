import random
import numpy as np


def gen_data(n=10000):
    gx = []
    gy = []
    for i in range(n):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        c = (a+b)/200
        gx.append([a/100, b/100])
        gy.append([c])
    return np.array(gx), np.array(gy)


x, y = gen_data(10000)
# test_x, test_y = gen_data(100)
test_x, test_y = np.array([[0.1, 0.9]]), np.array([[0.5]])
# print(np.shape(x), np.shape(y))
# print(test_x.T)
