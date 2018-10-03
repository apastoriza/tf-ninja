import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from random import randint
import numpy as np

from tfninja.resources import config
from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

BATCH_SIZE = 100
TRAINING_EPOCHS = 1000
EXPECTED_ACCURACY = 0.92

# About NMIST database
NMIST_IMAGE_PX_WIDTH = 28
NMIST_IMAGE_PX_HEIGHT = 28
TYPES_TO_PREDICT = 10

X_image = tf.placeholder(tf.float32, [None, NMIST_IMAGE_PX_WIDTH * NMIST_IMAGE_PX_HEIGHT], name='input')
Y_probabilities = tf.placeholder(tf.float32, [None, TYPES_TO_PREDICT])
W = tf.Variable(tf.zeros([NMIST_IMAGE_PX_WIDTH * NMIST_IMAGE_PX_HEIGHT, TYPES_TO_PREDICT]))
bias_tensor = tf.Variable(tf.zeros([TYPES_TO_PREDICT]))
XX_flatten_images = tf.reshape(X_image, [-1, NMIST_IMAGE_PX_WIDTH * NMIST_IMAGE_PX_HEIGHT])

evidence = tf.matmul(XX_flatten_images, W) + bias_tensor
Y = tf.nn.softmax(evidence, name='output')

softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_probabilities, logits=Y)
cross_entropy = tf.reduce_mean(softmax_cross_entropy_with_logits)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_probabilities, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


def gather_data():
    train_dir = config.paths['dir'] + 'data'
    data_sets = input_data.read_data_sets(train_dir=train_dir, one_hot=True)
    return data_sets


def setup_tensor_board(session):
    logs_path = config.paths['dir'] + '/logs/tfninja_softmax101'

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
        epoch = 0
        accuracy_value = 0.0
        data_sets = gather_data()
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
                logger.info('Expected accuracy: %s', accuracy_value)

            epoch += 1

        logger.info('Epoch: %s', epoch)
        logger.info('Expected accuracy: %s', accuracy_value)
        logger.info('-------TRAINING DONE-------')
        predict_numbers(session, data_sets.test)


def predict_numbers(session, test_data_set):
    trials = 100
    rights = 0
    for _ in range(trials):
        num = randint(0, test_data_set.images.shape[0])
        img = test_data_set.images[num]
        classification = session.run(tf.argmax(Y, 1), feed_dict={
            X_image: [img]
        })

        if classification[0] == np.argmax(test_data_set.labels[num]):
            rights += 1
            logger.debug('Neural Network predicted %s', classification[0])
            logger.debug('Real label is: %s', np.argmax(test_data_set.labels[num]))
        else:
            logger.warn('Neural Network predicted %s', classification[0])
            logger.warn('Real label is: %s', np.argmax(test_data_set.labels[num]))

    logger.info('Real accuracy: %s', (rights / trials))


if __name__ == '__main__':
    run_session()
