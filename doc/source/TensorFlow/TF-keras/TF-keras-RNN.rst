.. _header-n0:

TF-keras-RNN
============

.. _header-n3:

Keras RNN API
-------------

-  易用: 内置 RNN 神经网路层 APIs

   -  ``tf.keras.layers.RNN``

   -  ``tf.kersa.layers.LSTM``

   -  ``tf.keras.layers.GRU``

-  易自定义开发

   -  可以使用自定义自己的 RNN 单元层，并将其与通用的
      ``tf.keras.layers.RNN`` 一起使用

.. code:: python

   from __future__ import absolute_import, division, print_function, unicode_literals
   import collections
   import matplotlib.pyplot as plt 
   import numpy as np 
   import tensorflow as tf 
   from tensorflow.keras import layers

.. code:: python

   model = tf.keras.Sequential()
   model.add(layers.Embedding(input_dim = 1000, output_dim = 64))
   model.add(layers.LSTM(128))
   model.add(layers.Dense(10, activation = "softmax"))
   model.summary()

.. code:: python

   model = tf.keras.Sequential()
   model.add(layers.Embedding(input_dim = 1000, output_dim = 64))
   model.add(layers.GRU(256, return_sequences = True))
   model.add(layers.SimpleRNN(128))
   model.add(layers.Dense(10, activation = "softmax"))
   model.summary()

.. _header-n26:

RNN with list/dict inputs, or nested inputs
-------------------------------------------
