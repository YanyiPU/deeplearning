import numpy as np
import tensorflow as tf


def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label