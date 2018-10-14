# coding=utf-8

import tensorflow as tf
import numpy as np

from tfninja.tf103 import lenet101_softmax_model
from tfninja.resources import mnist_input_data
from tfninja.utils import loggerfactory
from tfninja.utils import tensorfactory

BATCH_SIZE = 128
TEST_SIZE = 256
NUM_CLASSES = 10
LEARNING_RATE = 0.001
DECAY = 0.9
EXPECTED_ACCURACY = 0.99
TRAINING_EPOCHS = 1000

# About MNIST database
IMAGE_PX_WIDTH = 28
IMAGE_PX_HEIGHT = 28
IMAGE_SIZE = IMAGE_PX_WIDTH * IMAGE_PX_HEIGHT

# Features map
FEATURES_MAP_CONV_1 = 32
FEATURES_MAP_CONV_2 = 64
FEATURES_MAP_CONV_3 = 128

LAYER_NEURONS_1 = FEATURES_MAP_CONV_3 * 4 * 4
LAYER_NEURONS_2 = 512

mnist = mnist_input_data.gather_data()

train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

train_x_reshape = train_x.reshape(-1, IMAGE_PX_WIDTH, IMAGE_PX_HEIGHT, 1)
test_x_reshape = test_x.reshape(-1, IMAGE_PX_WIDTH, IMAGE_PX_HEIGHT, 1)

X = tf.placeholder('float', [None, IMAGE_PX_WIDTH, IMAGE_PX_HEIGHT, 1])
Y = tf.placeholder('float', [None, NUM_CLASSES])

W_conv_layer_1 = tensorfactory.random_normal_variable([
    3, 3, 1, FEATURES_MAP_CONV_1
], 'weight_conv_layer_1')

W_conv_layer_2 = tensorfactory.random_normal_variable([
    3, 3, FEATURES_MAP_CONV_1, FEATURES_MAP_CONV_2
], 'weight_conv_layer_2')

W_conv_layer_3 = tensorfactory.random_normal_variable([
    3, 3, FEATURES_MAP_CONV_2, FEATURES_MAP_CONV_3
], 'weight_conv_layer_3')

W_layer_4 = tensorfactory.random_normal_variable([
    LAYER_NEURONS_1, LAYER_NEURONS_2
], 'weight_layer_4')

W_layer_output = tensorfactory.random_normal_variable([
    LAYER_NEURONS_2, NUM_CLASSES
], 'weight_layer_output')

keep_prob_conv = tf.placeholder('float')
keep_prob_hidden = tf.placeholder('float')
py_x = lenet101_softmax_model.model(X,
                                    W_conv_layer_1, W_conv_layer_2, W_conv_layer_3, W_layer_4, W_layer_output,
                                    keep_prob_conv, keep_prob_hidden
                                    )

softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y)
cost = tf.reduce_mean(softmax_cross_entropy_with_logits)
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY).minimize(cost)
predict_op = tf.argmax(py_x, 1)


def run_session():
    with tf.Session() as session:
        logger = loggerfactory.get_logger(__name__)
        session.run(tf.global_variables_initializer())
        mean = 0
        epoch = 0
        while (epoch < TRAINING_EPOCHS) and (mean <= EXPECTED_ACCURACY):
            training_batch = zip(
                range(0, len(train_x_reshape), BATCH_SIZE),
                range(BATCH_SIZE, len(train_x_reshape) + 1, BATCH_SIZE)
            )
            for start, end in training_batch:
                session.run(optimizer, feed_dict={
                    X: train_x_reshape[start:end],
                    Y: train_y[start:end],
                    keep_prob_conv: 0.8,
                    keep_prob_hidden: 0.5
                })

            test_indices = np.arange(len(test_x_reshape))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:TEST_SIZE]

            session_prediction = session.run(predict_op, feed_dict={
                X: test_x_reshape[test_indices],
                Y: test_y[test_indices],
                keep_prob_conv: 1.0,
                keep_prob_hidden: 1.0
            })

            mean = np.mean(np.argmax(test_y[test_indices], axis=1) == session_prediction)
            logger.info('Epoch: %s - accuracy: %s', epoch, mean)
            epoch += 1


if __name__ == '__main__':
    run_session()
