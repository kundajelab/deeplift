import tensorflow as tf

def placeholder(shape, name):
    return tf.placeholder(tf.float32, shape=shape, name=name)
