# coding=utf-8

import numpy as np
import tensorflow as tf

from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

scalar = tf.constant(100)
vector = tf.constant([1, 2, 3, 4, 5])
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
cube_matrix = tf.constant([
    [
        [1], [2], [3]
    ], [
        [4], [5], [6]
    ], [
        [7], [8], [9]
    ]
])

logger.info('scalar    (native): %s', scalar.get_shape())
logger.info('vector    (native): %s', vector.get_shape())
logger.info('matrix    (native): %s', matrix.get_shape())
logger.info('cube      (native): %s', cube_matrix.get_shape())

# create a tf.constant() from numpy array
np_vector = np.array([6, 7, 8, 9, 10])
vector2 = tf.constant(np_vector)
logger.info('vector    (numpy ): %s', vector2.get_shape())

# another way to create a tensor from numpy array
np_3d = np.array([
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ], [
        [9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]
    ], [
        [18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]
    ]
])
tensor_3d = tf.convert_to_tensor(np_3d, dtype=tf.float64)
logger.info('tensor_3d (numpy ): %s', vector2.get_shape())
