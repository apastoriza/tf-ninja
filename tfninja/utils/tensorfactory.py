# coding=utf-8

import tensorflow as tf

DEFAULT_STD_DEV = 0.01


def random_normal_variable(shape, variable_name, std_dev=DEFAULT_STD_DEV):
    variable = tf.Variable(tf.random_normal(shape, stddev=std_dev), name=variable_name)
    return variable
