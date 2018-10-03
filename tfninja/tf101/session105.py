# coding=utf-8

# single neuron and TensorBoard
# --(weight)--(input)-->[output=f(input, weight)]--(output)-->

import tensorflow as tf

from tfninja.resources import config
from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

MAX_ITERATIONS = 100000
TARGET_RESULT = 0.0000000001
DESCENT_OPTIMIZER = 0.001

# an input value who stimulates the neuron. It will be a constant to make it simple
INPUT_VALUE = tf.constant(0.5, name='INPUT_VALUE')

# the value whe expect to get
EXPECTED_OUTPUT = tf.constant(0.0, name='EXPECTED_OUTPUT')

# a weight, multiplied by the input to provide the output of the neuron
weight = tf.Variable(1.0, name='weight')

# the function model
model = tf.multiply(INPUT_VALUE, weight, 'model')

loss_function = tf.pow((EXPECTED_OUTPUT - model), 2, name='loss_function')

optimizer = tf.train.GradientDescentOptimizer(DESCENT_OPTIMIZER).minimize(loss_function)


def setup_tensor_board(session):
    log_dir = config.paths['dir'] + '/logs/tfninja_session105'

    # define the parameters to be displayed in TensorBoard
    for value in [INPUT_VALUE, EXPECTED_OUTPUT, weight, model, loss_function]:
        tf.summary.scalar(value.op.name, value)

    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, session.graph)
    return summaries, summary_writer


def run_session():
    with tf.Session() as session:
        summaries, summary_writer = setup_tensor_board(session)
        session.run(tf.global_variables_initializer())

        i = 0
        result = session.run(weight)
        logger.debug('Adjusted weight(%s): %s', i, result)

        while (i < MAX_ITERATIONS) and (result >= TARGET_RESULT):
            current_summary = session.run(summaries)
            summary_writer.add_summary(current_summary, i)
            session.run(optimizer)
            result = session.run(weight)
            logger.debug('Adjusted weight(%s): %s', i, result)
            i += 1


if __name__ == '__main__':
    run_session()
