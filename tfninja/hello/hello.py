import tensorflow as tf
from tfninja.utils import loggerfactory

logger = loggerfactory.get_logger(__name__)

hello = tf.constant("hello ninjas!!")
session = tf.Session()
logger.info(session.run(hello))
