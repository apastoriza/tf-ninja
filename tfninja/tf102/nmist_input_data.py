# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data

from tfninja.resources import config


def gather_data():
    train_dir = config.paths['dir'] + 'data'
    data_sets = input_data.read_data_sets(train_dir=train_dir, one_hot=True)
    return data_sets

