
Keras 模型编译
==============

.. code:: python

   model.compile(loss,
                 optimizer,
                 metrics)


1. Losses
---------

   -  Loss Function

   -  Objective Function

   -  Optimization score Function

**(1) Keras loss functions**

**回归：**

.. code:: python

   from keras import losses

   # 回归 
   from keras.losses import mean_squared_error
   from keras.losses import mean_absolute_error
   from keras.losses import mean_absolute_percentage_error
   from keras.losses import mean_squared_logarithmic_error
   from keras.losses import squared_hinge
   from keras.losses import hinge
   from keras.losses import categorical_hinge
   from keras.losses import logcosh

.. code:: python

   model.Compile(loss = ["mse", "MSE", mean_squared_error], 
   			  optimizer, 
   			  metircs)
   model.Compile(loss = ["mae", "MAE", mean_absolute_error], 
   			  optimizer, 
   			  metircs)
   model.Compile(loss = ["mape", "MAPE", mean_absolute_percentage_error], 
   			  optimizer, 
   			  metircs)
   model.Compile(loss = ["msle", "MLSE", mean_squared_logarithmic_error], 
   			  optimizer, 
   			  metircs)

**分类：**

.. code:: python

   # 分类
   from keras.losses import categorical_crossentropy
   from keras.losses import sparse_categorical_crossentropy
   from keras.losses import binary_crossentropy
   from keras.losses import kullback_leibler_divergence
   from keras.losses import poisson
   from keras.losses import cosine_proximity

.. code:: python

   model.Compile(loss = ["kld", "KLD", kullback_leibler_divergence], 
   			  optimizer, 
   			  metircs)
   model.Compile(loss = ["cosine", cosine_proximity], 
   			  optimizer, 
   			  metircs)


2. Metrics
----------

   -  Metric 是一个评估模型表现的函数

   -  Metric 函数类似于一个损失函数，只不过模型评估返回的 metric
      不用来训练模型，因此，可以使用任何损失函数当做一个 metric 函数使用


2.1 Keras metrics
~~~~~~~~~~~~~~~~~

Keras API:

.. code:: python

   from keras import metrics
   from keras.metrics import binary_accuracy
   from keras.metrics import categorical_accuracy
   from keras.metrics import sparse_categorical_accuracy
   from keras.metrics import top_k_categorical_accuracy
   from keras.metrics import sparse_top_k_categorical_accuracy
   from keras.metrics import mae

   from keras.losses import mean_squared_error
   from keras.losses import mean_absolute_error
   from keras.losses import mean_absolute_percentage_error
   from keras.losses import mean_squared_logarithmic_error
   from keras.losses import squared_hinge
   from keras.losses import hinge
   from keras.losses import categorical_hinge
   from keras.losses import logcosh
   from keras.losses import categorical_crossentropy
   from keras.losses import sparse_categorical_crossentropy
   from keras.losses import binary_crossentropy
   from keras.losses import kullback_leibler_divergence
   from keras.losses import poisson
   from keras.losses import cosine_proximity

Metrics Name:

.. code:: python

   metrics = ["acc", "accuracy"]


2.3 自定义 metrics
~~~~~~~~~~~~~~~~~~

.. code:: python

   import keras.backend as K

   def mean_pred(y_true, y_pred):
   	return K.mean(y_pred)

   model.compile(optimizers = "rmsprop",
   			  loss = "binary_accuracy",
   			  metrics = ["accuracy", mean_pred])


3. Optimizers
-------------


3.1 Keras optimizder 的使用方式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(1) ``keras.optimizers`` 和 ``optimizer`` 参数

.. code:: python

   from keras import optimizers

   # 编译模型
   sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
   model.compile(loss = "mean_squared_error",
                 optimizer = sgd)

(2) ``optimizer`` 参数

.. code:: python

   # 编译模型
   model.compile(loss = "mean_squared_error",
                 optimizer = "sgd")



3.2 Keras optimizers 的共有参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  control gradient clipping

   -  ``clipnorm``

   -  ``clipvalue``

.. code:: python

   from keras import optimizers

   # All parameter gradients will be clipped to
   # a maximum norm of 1.
   sgd = optimizers.SGD(lr = 0.01, clipnorm = 1)

   # All parameter gradients will be clipped to
   # a maximum value of 0.5 and
   # a minimum value of -0.5.
   sgd = optimizers.SGD(lr = 0.01, clipvalue = 0.5)



3.3 Keras Optimizers
~~~~~~~~~~~~~~~~~~~~

-  SGD

-  RMSprop

-  Adagrad

-  Adadelta

-  Adam

-  Adamax

-  Nadam

.. code:: python

   from keras import optimizers

   sgd = optimizers.SGD(lr = 0.01)
   model.compile(loss, optimizer = sgd)
   # or
   model.compile(loss, optimizer = "sgd")

   rmsprop = optimizers.RMSprop(lr = 0.001)
   model.compile(loss, optimizer = rmsprop)
   # or
   model.compile(loss, optimizer = "rmsprop")

   adagrad = optimizers.Adagrad(lr = 0.01)
   model.compile(loss, optimizer = adagrad)
   # or
   model.compile(loss, optimizer = "adagrad")

   adadelta = optimizers.Adadelta(lr = 1.0)
   model.compile(loss, optimizer = adadelta)
   # or
   model.compile(loss, optimizer = "adadelta")

   adam = optimizers.Adam(lr = 0.001)
   model.compile(loss, optimizer = adam)
   # or
   model.compile(loss, optimizer = "adam")

   adamax = optimizers.Adamax(lr = 0.02)
   model.compile(loss, optimizer = adamax)
   # or
   model.compile(loss, optimizer = "adamax")

   nadam = optimizers.Nadam(lr = 0.002)
   model.compile(loss, optimizer = nadam)
   # or
   model.compile(loss, optimizer = "nadam")
