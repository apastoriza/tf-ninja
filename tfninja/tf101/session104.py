# coding=utf-8

import numpy as np
import tensorflow as tf

from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

rows = 3
cols = 2
x = tf.placeholder(tf.float32, name='random_matrix', shape=(rows, cols))
add_operation = tf.add(x, x)


def run_session():
    data = np.random.rand(rows, cols)
    logger.info('\ndata:\n %s', data)
    with tf.Session() as session:
        result = session.run(add_operation, feed_dict={
            x: data
        })

        logger.info('\nadd result:\n %s', result)


if __name__ == '__main__':
    run_session()
