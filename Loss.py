from Sampling import Sampling
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Loss:
    def reconstruction(self, x1, x2):
        loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x1, x2), axis=(1, 2)))
        return loss

    def kullback_leibler(self, z_mean, z_log_var):
        loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return loss