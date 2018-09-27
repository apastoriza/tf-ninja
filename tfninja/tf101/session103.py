# coding=utf-8

import tensorflow as tf

from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

CONSTANT_A = tf.constant([100.0])
CONSTANT_B = tf.constant([200.0])
CONSTANT_C = tf.constant([10.0])
add_operation = tf.add(CONSTANT_A, CONSTANT_B)
multiply_operation = tf.multiply(CONSTANT_A, CONSTANT_C)


def run_session():

    with tf.Session() as session:
        result = session.run([add_operation, multiply_operation])
        logger.info(result)


if __name__ == '__main__':
    run_session()
