import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras