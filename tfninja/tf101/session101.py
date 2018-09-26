# coding=utf-8

import tensorflow as tf
from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)
with tf.Session() as session:
    # create a sensor x
    x = tf.placeholder(
        tf.float32,  # data type
        [1],  # tensor shape (one dimension)
        name='x'
    )

    # create a constant z
    z = tf.constant(2.0)

    y = x * z
    x_in = [100]

    y_output = session.run(y, {
        x: x_in
    })
logger.info(y_output)
