#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import mnist


(train_data, train_labels), (test_data, test_labesl) = mnist.load_data()
print(train_data.shape)
print(len(train_data))
print(train_labels)

print(test_data.shape)
print(len(test_labesl))
print(test_labesl)
