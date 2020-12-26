import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


"""
y = ax + b
"""


class Linear(tf.keras.Model):
    """
    线性回归

    Args:
        tf ([type]): [description]
    """
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units = 1,
            activation = None,
            use_bias = True,
            kernel_initializer = tf.zeros_initializer(),
            bias_initializer = tf.zeros_initializer()
        )
    
    def call(self, input):
        output = self.dense(input)
        return output





if __name__ == "__main__":
    num_epochs = 100
    batch_size = np.nan
    learning_rate = 1e-3

    # data
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    # model
    model = Linear()

    # optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
    
    # model training
    for i in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_sum(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)

        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

    print(model.variables)
