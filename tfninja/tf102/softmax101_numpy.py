# coding=utf-8

import numpy as np


def run_softmax():
    y = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3]
    softmax_y = np.exp(y) / np.sum(np.exp(y))
    print(y)
    print(softmax_y)

    z = [0.1, 0.2, 0.3, 4.0, 0.1, 0.2, 0.3]
    softmax_z = np.exp(z) / np.sum(np.exp(z))
    print(z)
    print(softmax_z)


if __name__ == '__main__':
    run_softmax()
