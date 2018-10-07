# coding=utf-8

import math


def run_softmax():
    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    print([round(i, 3) for i in z])
    z_exp = [math.exp(i) for i in z]
    print([round(i, 3) for i in z_exp])
    sum_z_exp = sum(z_exp)
    # print(round(sum_z_exp, 2))
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    print(softmax)


if __name__ == '__main__':
    run_softmax()
