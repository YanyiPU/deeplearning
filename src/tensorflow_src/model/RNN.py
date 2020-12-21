import numpy as np
import tensorflow as tf


class RNN(tf.model.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cell = tf.keras.layers.LSTMCell(units = 256)
        self.dense = tf.keras.layers.Dense(units = self.num_chars)
    
    def call(self, input, from_logits = False):
        # [batch_size, seq_length, num_cahrs]
        inputs = tf.one_hot(input, depth = self.num_chars)
        state = self.cell.get_initial_state(batch_size = self.batch_size, dtype = tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(input[:, t, :], state)
            logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)
        output = None
        return output


if __name__ == "__main__":
    num_epochs = 1000
    seq_length = 40
    batch_size = 50
    learning_rate = 1e-3

    # data
    data_loader = DataLoader()

    # model
    model = RNN(num_chars = len(data_loader.chars), batch_size = batch_size, seq_length = seq_length)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    # batches
    for batch_index in range(num_batches):
        


    # model training


    # metric



