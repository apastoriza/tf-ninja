# coding=utf-8

import tensorflow as tf
from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

value = tf.Variable(0, name="value")
ONE = tf.constant(1)
new_value = tf.add(value, ONE)
update_value = tf.assign(value, new_value)


def run_session():
    # Only after running tf.global_variables_initializer() in a session will your variables hold the values you told
    # them to hold when you declare them (tf.Variable(tf.zeros(...)), tf.Variable(tf.random_normal(...)),...).
    initialize_var = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(initialize_var)
        output = session.run(value)
        logger.info('Variable value: %s', output)
        for _ in range(10):
            # execute assign operation
            session.run(update_value)
            # Variable value
            output = session.run(value)
            logger.info('Variable value: %s', output)


if __name__ == '__main__':
    run_session()
