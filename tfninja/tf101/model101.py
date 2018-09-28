# coding=utf-8

import numpy as np
import tensorflow as tf

from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

SCALAR = tf.constant(100)
VECTOR = tf.constant([1, 2, 3, 4, 5])
MATRIX = tf.constant([[1, 2, 3], [4, 5, 6]])
CUBE_MATRIX = tf.constant([
    [
        [1], [2], [3]
    ], [
        [4], [5], [6]
    ], [
        [7], [8], [9]
    ]
])

logger.info('scalar    (native): %s', SCALAR.get_shape())
logger.info('vector    (native): %s', VECTOR.get_shape())
logger.info('matrix    (native): %s', MATRIX.get_shape())
logger.info('cube      (native): %s', CUBE_MATRIX.get_shape())

# create a tf.constant() from numpy array
np_vector = np.array([6, 7, 8, 9, 10])
VECTOR2 = tf.constant(np_vector)
logger.info('vector    (numpy ): %s', VECTOR2.get_shape())

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
TENSOR_3D = tf.convert_to_tensor(np_3d, dtype=tf.float64)
logger.info('tensor_3d (numpy ): %s', TENSOR_3D.get_shape())
