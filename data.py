import random
import numpy as np


def gen_data(n=10000):
    gx = []
    gy = []
    for i in range(n):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        gx.append([a/100, b/100])
        gy.append([(a+2*b)/300])
    return np.array(gx, dtype="float32"), np.array(gy)


x, y = gen_data(5000)
# test_x, test_y = gen_data(600)
test_x, test_y = np.array([[0.2, 0.5]]), np.array([[0.4]])
