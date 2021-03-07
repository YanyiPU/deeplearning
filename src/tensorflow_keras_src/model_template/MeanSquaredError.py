import numpy as np
import tensorflow as tf


class MeanSquaredError(tf.keras.losses.Loss):
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
