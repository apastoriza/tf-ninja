import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from random import randint
import numpy as np

from tfninja.resources import config
from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

BATCH_SIZE = 100
TRAINING_EPOCHS = 10000
EXPECTED_ACCURACY = 0.90

# About MNIST database
MNIST_IMAGE_PX_WIDTH = 28
MNIST_IMAGE_PX_HEIGHT = 28
TYPES_TO_PREDICT = 10

mnist = input_data.read_data_sets('data', one_hot=True)

X = tf.placeholder(tf.float32, [None, MNIST_IMAGE_PX_WIDTH * MNIST_IMAGE_PX_HEIGHT], name='input')
Y_ = tf.placeholder(tf.float32, [None, TYPES_TO_PREDICT])
W = tf.Variable(tf.zeros([MNIST_IMAGE_PX_WIDTH * MNIST_IMAGE_PX_HEIGHT, TYPES_TO_PREDICT]))
b = tf.Variable(tf.zeros([TYPES_TO_PREDICT]))
XX = tf.reshape(X, [-1, MNIST_IMAGE_PX_WIDTH * MNIST_IMAGE_PX_HEIGHT])

Y = tf.nn.softmax(tf.matmul(XX, W) + b, name='output')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_, logits=Y))
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


def setup_tensor_board(session):
    logs_path = config.paths['dir'] + '/logs/tfninja_mnist101'

    tf.summary.scalar('cost', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summaries = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(logs_path, graph=session.graph)

    return summaries, summary_writer


def run_session():
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        summaries, summary_writer = setup_tensor_board(session)

        epoch = 0;
        accuracy_value = 0.0
        while (epoch < TRAINING_EPOCHS) and (accuracy_value <= EXPECTED_ACCURACY):
            batch_count = int(mnist.train.num_examples / BATCH_SIZE)
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                _, summary = session.run([train_step, summaries], feed_dict={
                    X: batch_x, Y_: batch_y
                })
                summary_writer.add_summary(summary, epoch * batch_count + i)
            accuracy_value = accuracy.eval(feed_dict={
                X: mnist.test.images,
                Y_: mnist.test.labels
            })
            logger.info('Epoch: %s', epoch)
            logger.info('Accuracy: %s', accuracy_value)
            epoch += 1

        logger.info('-------done-------')
        predict_numbers(session)
        save_session(session)


def predict_numbers(session):
    for _ in range(100):
        num = randint(0, mnist.test.images.shape[0])
        img = mnist.test.images[num]
        classification = session.run(tf.argmax(Y, 1), feed_dict={X: [img]})
        logger.info('Neural Network predicted %s', classification[0])
        logger.info('Real label is: %s', np.argmax(mnist.test.labels[num]))


def save_session(session):
    save_path = config.paths['dir'] + '/data/tfninja_mnist101.ckpt'
    saver = tf.train.Saver()
    session_saver = saver.save(session, save_path)
    logger.info('Model saved to %s', session_saver)


if __name__ == '__main__':
    run_session()
