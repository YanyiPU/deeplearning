

# Keras-CIFAR10-CNN_Capsule


"""
This example trains a simple CNN-Capsule Network on the CIFAR10 data set.

Without Data Augmentation:
It gets to 75% validation accuracy in 10 epochs, 79% after 15 epochs,
and overfitting after 20 epochs

With Data Augmentation:
It gets to 75% validation accuracy in 10 epochs, 79% after 15 epochs,
and 83% after 30 epochs.

The highest achieved validation accuracy is 83.79% after 50 epochs.
This is a fast implementation that takes just 20s/epoch on a GTX 1070 GPU.

The paper "Dynamic Routing Between Capsules": https://arxiv.org/abs/1710.09829
"""

from __future__ import print_function

import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import activation
from keras import layers
from keras import utils
from keras.models import Model


def squash(x, axis = -1):
"""The Squashing Function.
    The nonlinear activation function used in Capsule Network
    # Arguments
        x: Input Tensor.
        axis: Integer axis along which the squashing function is to be applied.
    # Returns
        Tensor with scaled value of the input tensor
    """
s_squared_norm = K.sum(K.square(x), axis, keepdims = True) + K.epsilon()
scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
return scale * x

def margin_loss(y_true, y_pred):
"""Margin loss
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + \
                lamb * (1 - y_true) * K.square(K.relu(y_pred - margin)), 
                axis = 1)
