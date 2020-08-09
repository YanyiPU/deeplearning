.. _header-n0:

TF-keras-custom
===============

.. code:: python

   from __future__ import absolute_import, division, print_function, unicode_literals
   import tensorflow as tf 
   tf.keras.backend.clear_session()

.. _header-n5:

1. Layer class
--------------

-  ``Layers``

   -  state

      -  layer weights

      -  ``__init__()``

   -  computation

      -  transformation from inputs to outputs(a "call", the layer's
         forward pass)

      -  ``call()``

Example:

.. code:: python

   from tensorflow.keras import layers

   class Linear(layers.Layer):
       """
       densely-connected layer.
           state: w, b
       """
       def __init__(self, units = 32, input_dim = 32):
           super(Linear, self).__init__()
           w_init = tf.random_normal_initializer()
           self.w = tf.Variable(initial_value = w_init(shape = (input_dim, units), dtype = "float32"),
                                trainable = True)
           b_init = tf.zeros_initializer()
           self.b = tf.Variable(initial_value = b_init(shape = (units,), dtype = "float32"),
                                trainable = True)
       def call(self, inputs):
           y = tf.matmul(inputs, self.w) + self.b
           return y

   x = tf.ones((2, 2))
   linear_layer = Linear(4, 2)
   assert linear_layer.weights == [linear_layer.w, linear_layer.b]
   y = linear_layer(x)
   print(y)

.. code:: python

   from tensorflow.keras import layers

   class Linear(layers.Layer):
       """
       Using `add_weight` method.
       densely-connected layer.
           state: w, b
       """
       def __init__(self, units = 32, input_dim = 32):
           super(Linear, self).__init__()
           self.w = self.add_weight(shape = (input_dim, units),
                                    initializer = "random_normal",
                                    trainable = True)
           self.b = self.add_weight(shape = (units,),
                                    initializer = "zeros",
                                    trainable = True)
       def call(self, inputs):
           y = tf.matmul(inputs, self.w) + self.b
           return y

   x = tf.ones((2, 2))
   linear_layer = Linear(4, 2)
   assert linear_layer.weights == [linear_layer.w, linear_layer.b]
   y = linear_layer(x)
   print(y)

.. code:: python

   from tensorflow.keras import layers

   class ComputeSum(layers.Layer):
       """
       non-trainable weights.
       Weights are meant not to be taken into account during backpropagation 
       when you are training the layer.
       """
       def __init__(self, input_dim):
           super(ComputeSum, self).__init__()
           self.total = tf.Variable(initial_value = tf.zeros((input_dim,)),
                                    trainable = False)
           # total = tf.zeros_initializer()
           # self.total = tf.Variable(initial_value = total(shape = (input_dim,), dtype = "float32"),
           #                          trainable = False)

       def call(self, inputs):
           self.total.assign_add(tf.reduce_sum(inputs, axis = 0))
           return self.total

   x = tf.ones((2, 2))
   my_sum = ComputeSum(2)
   y = my_sum(x)
   print(y.numpy())
   y = my_sum(x)
   print(y.numpy())

   print("weights:", len(my_sum.weights))
   print("non-trainable weights:", len(my_sum.non_trainable_weights))
   print("trainable_weights:", my_sum.trainable_weights)

.. code:: python

   class Linear(layers.Layer):
       """
       将权重创建推迟到知道输入的形状为止
       """
       def __init__(self, units = 32):
           super(Linear, self).__init__()
           self.units = units

       def build(self, input_shape):
           self.w = self.add_weight(shape = (input_shape[-1], self.units), 
                                    initializer = "random_normal",
                                    trainable = True)
           self.b = self.add_weight(shape = (self.units,),
                                    initializer = "rando_normal",
                                    trainable = True)

       def call(self, inputs):
           y = tf.matmul(inputs, self.w) + self.b

   linear_layer = Linear(32)
   y = linear_layer(x)

.. code:: python

   class MLPBlock(layers.Layer):
       """
       层是可递归组合
       """
       def __init__(self):
           super(MLPBlock, self).__init__()
           self.linear_1 = Linear(32)
           self.linear_2 = Linear(32)
           self.linear_3 = Linear(1)

       def call(self, inputs):
           x = self.linear_1(inputs)
           x = tf.nn.relu(x)
           x = self.linear_2(x)
           x = tf.nn.relu(x)
           y = self.linear_3(x)
           return y

   mlp = MLPBlock()
   y = mlp(tf.ones(shape = (3, 64)))
   print("weights:", len(mlp.weights))
   print("trainable weights:", len(mlp.trainable_weights))

.. _header-n36:

2. 构建模型
-----------
