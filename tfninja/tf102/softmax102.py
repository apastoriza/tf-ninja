# coding=utf-8

import numpy as np
import tensorflow as tf

from random import randint

from tfninja.resources import config
from tfninja.resources import mnist_input_data
from tfninja.utils import loggerfactory
from tfninja.utils import time

logger = loggerfactory.get_logger(__name__)

BATCH_SIZE = 100
TRAINING_EPOCHS = 1000
EXPECTED_ACCURACY = 0.90
LEARNING_RATE = 0.005
LAYER_NEURONS = 10

# About MNIST database
IMAGE_PX_WIDTH = 28
IMAGE_PX_HEIGHT = 28


X_image = tf.placeholder(tf.float32, [None, IMAGE_PX_WIDTH * IMAGE_PX_HEIGHT], name='input')
Y_probabilities = tf.placeholder(tf.float32, [None, LAYER_NEURONS])
W = tf.Variable(tf.zeros([IMAGE_PX_WIDTH * IMAGE_PX_HEIGHT, LAYER_NEURONS]))
bias_tensor = tf.Variable(tf.zeros([LAYER_NEURONS]))
XX_flatten_images = tf.reshape(X_image, [-1, IMAGE_PX_WIDTH * IMAGE_PX_HEIGHT])

evidence = tf.matmul(XX_flatten_images, W) + bias_tensor
Y = tf.nn.softmax(evidence, name='output')

softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_probabilities, logits=Y)
cross_entropy = tf.reduce_mean(softmax_cross_entropy_with_logits)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_probabilities, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# try to switch between gradient and adam optimizer to see the effect
# train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)


def setup_tensor_board(session):
    logs_path = config.paths['dir'] + '/logs/tfninja_softmax102'

    tf.summary.scalar('cost', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summaries = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(logs_path, graph=session.graph)

    return summaries, summary_writer


def run_session():
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        summaries, summary_writer = setup_tensor_board(session)

        logger.info('-------TRAINING INIT-------')
        init_time_in_millis = time.current_time_in_millis()
        epoch = 0
        accuracy_value = 0.0
        data_sets = mnist_input_data.gather_data()
        while (epoch < TRAINING_EPOCHS) and (accuracy_value <= EXPECTED_ACCURACY):
            batch_count = int(data_sets.train.num_examples / BATCH_SIZE)
            for i in range(batch_count):
                batch_x, batch_y = data_sets.train.next_batch(BATCH_SIZE)
                _, summary = session.run([train_step, summaries], feed_dict={
                    X_image: batch_x,
                    Y_probabilities: batch_y
                })
                summary_writer.add_summary(summary, epoch * batch_count + i)

            accuracy_value = accuracy.eval(feed_dict={
                X_image: data_sets.test.images,
                Y_probabilities: data_sets.test.labels
            })

            if epoch % 10 == 0:
                logger.info('Epoch: %s', epoch)
                logger.info('Current accuracy: %s', accuracy_value)

            epoch += 1

        end_time_in_millis = time.current_time_in_millis()
        logger.info('Epoch: %s', epoch)
        logger.info('-------TRAINING DONE-------')
        logger.info('Total time: %s millis', (end_time_in_millis - init_time_in_millis))
        logger.info('Expected accuracy: %s', accuracy_value)
        predict_numbers(session, data_sets.test)


def predict_numbers(session, test_data_set):
    trials = 1000
    rights = 0
    for _ in range(trials):
        num = randint(0, test_data_set.images.shape[0])
        img = test_data_set.images[num]
        classification = session.run(tf.argmax(Y, 1), feed_dict={
            X_image: [img]
        })

        if classification[0] == np.argmax(test_data_set.labels[num]):
            rights += 1
            # logger.debug('Neural Network predicted %s', classification[0])
            # logger.debug('Real label is: %s', np.argmax(test_data_set.labels[num]))
        else:
            logger.error('Neural Network predicted %s', classification[0])
            logger.error('Real label is: %s', np.argmax(test_data_set.labels[num]))

    logger.info('Real accuracy: %s/%s = %s', rights, trials, (rights / trials))


if __name__ == '__main__':
    run_session()
