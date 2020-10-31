#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import models, layers
from keras.datasets import boston_housing
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical


(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
print(train_data.shape)
print(len(train_data))
print(train_labels)

print(test_data.shape)
print(len(test_labels))
print(test_labels)