.. _header-n0:

TensorFlow-MNIST-CNN
====================

.. code:: python

   import tensorflow as tf
   from tensorflow.examples.tutorials.mnist
   import input_data

.. _header-n4:

data
----

.. code:: python

   mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

.. _header-n6:

1.简易神经网络
--------------

.. code:: python

   sess = tf.InteractiveSession()
