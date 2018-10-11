# coding=utf-8

import tensorflow as tf

PADDING_TYPE = 'SAME'
IMAGE_NUMBER = 1
INPUT_CHANNEL = 1
KERNEL_SIZE = [1, 2, 2, 1]


def model(x, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    conv1 = _create_conv_layer_dropout(x, w1, p_keep_conv)
    conv2 = _create_conv_layer_dropout(conv1, w2, p_keep_conv)
    conv3 = _create_conv_layer_relu(conv2, w3)

    fully_connected_layer = tf.nn.max_pool(conv3, ksize=KERNEL_SIZE, strides=[
        IMAGE_NUMBER, 2, 2, INPUT_CHANNEL
    ], padding=PADDING_TYPE)

    fully_connected_layer = tf.reshape(fully_connected_layer, [-1, w4.get_shape().as_list()[0]])
    fully_connected_layer = tf.nn.dropout(fully_connected_layer, p_keep_conv)

    output_layer = tf.nn.relu(tf.matmul(fully_connected_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result


def _create_conv_layer_relu(input_data, weight):
    conv_layer_input = tf.nn.conv2d(input_data, weight, strides=[
        IMAGE_NUMBER, 1, 1, INPUT_CHANNEL
    ], padding=PADDING_TYPE)
    return tf.nn.relu(conv_layer_input)


def _create_conv_layer_dropout(input_data, weight, keep_prob):
    conv_layer_relu = _create_conv_layer_relu(input_data, weight)
    conv_layer_maxpool = tf.nn.max_pool(conv_layer_relu, ksize=KERNEL_SIZE, strides=[
        IMAGE_NUMBER, 2, 2, INPUT_CHANNEL
    ], padding=PADDING_TYPE)
    conv_layer_dropout = tf.nn.dropout(conv_layer_maxpool, keep_prob)
    return conv_layer_dropout
