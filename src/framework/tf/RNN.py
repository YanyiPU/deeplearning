import numpy as np
import tensorflow as tf


class RNN(tf.model.Model):

    def __init__(self):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units = 256)
        self.dense = tf.keras.layers.Dense(units = self.num_chars)
    
    def call(self, input, from_logits = False):
        output = None
        return output
