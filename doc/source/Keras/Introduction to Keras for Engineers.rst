.. _header-n0:

Keras 管网
===================================

.. code:: python

   import numpy as np 
   import tensorflow as tf 
   from tensorflow import keras

.. _header-n5:

介绍
----

-  How to prepare you data before training a model (by turning it into
   either NumPy arrays or ``tf.data.Dataset`` objects).

-  How to do data preprocessing, for instance feature normalization or
   vocabulary indexing.

-  How to build a model that turns your data into useful predictions,
   using the Keras Functional API.

-  How to train your model with the built-in Keras ``fit()`` method,
   while being mindful of checkpointing, metrics monitoring, and fault
   tolerance.

-  How to evaluate your model on a test data and how to use it for
   inference on new data.

-  How to customize what ``fit()`` does, for instance to build a GAN.

-  How to speed up training by leveraging multiple GPUs.

-  How to refine your model through hyperparameter tuning.
