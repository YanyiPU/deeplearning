.. _header-n0:

TF-keras
========

.. _header-n3:

1. 导入 ``tf.keras``
--------------------

.. code:: python

   # in Python
   from __future__ import absolute_import, division, print_function, unicode_literals
   import tensorflow as tf
   from tensorflow import keras

   print(tf.VERSION)
   print(tf.keras.__version__)

.. _header-n5:

2. 构建模型
-----------

.. _header-n6:

2.1 构建序列模型
~~~~~~~~~~~~~~~~

序列模型：

-  ``tf.keras.Sequential``

配置层：

-  ``tf.keras.layers``

   -  ``tf.keras.layers.Dense``

      -  ``activation``: 设置层的激活函数

         -  ``relu``

            -  ``tf.keras.activations.relu``

         -  ``sigmoid``

            -  ``tf.keras.activations.sigmoid``

         -  ``tanh``

            -  ``tf.tanh``

      -  ``kernel_initializer``: 创建层权重(核)的初始化方案

         -  "Glorot uniform"

         -  "orthogonal"

         -  tf.keras.initializers.Constant(2.0)

      -  ``bias_initializer``: 创建层权重(偏差)的初始化方案

         -  "Glorot uniform"

         -  "orthogonal"

         -  tf.keras.initializers.Constant(2.0)

      -  ``kernel_regularizer``: 应用层权重(核)的正则化方案

         -  tf.keras.regularizers.l1(0.01)

         -  tf.keras.regularizers.l2(0.01)

      -  ``bias_regularizer``: 应用层权重(偏差)的正则化方案

         -  tf.keras.regularizers.l1(0.01)

         -  tf.keras.regularizers.l2(0.01)

构建，编译模型：

-  ``tf.keras.Model.compile``

   -  ``optimizer``

      -  ``tf.keras.optimizers``

         -  ``tf.keras.optimizers.Adam``

         -  "adam"

         -  ``tf.keras.optimizers.SGD``

         -  "sgd"

   -  ``loss``

      -  ``tf.keras.losses``

      -  ``mse``

      -  ``categorical_crossentropy``

      -  ``binary_crossentropy``

   -  ``metrics``

      -  ``tf.keras.metrics``

      -  ``accuracy``

      -  ``mae``

   -  ``run_eagerly = True``

输入 Numpy 数据,训练模型：

   -  numpy

   -  ``tf.keras.Model.fit``

      -  ``epochs``:

         -  以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）

      -  ``batch_size``

         -  模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小

      -  ``validation_data``

         -  在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标

输入 ``tf.data`` 数据集,训练模型：

-  ``tf.data.Dataset``

-  ``tf.keras.Model.fit``

   -  ``epochs``

   -  ``batch_size``

   -  ``validation_data``

模型评估和预测

-  ``tf.keras.Model.evaluate``

-  ``tf.keras.Model.predict``

.. _header-n152:

2.2 构建复杂模型
~~~~~~~~~~~~~~~~

   -  tf.keras.Sequential 模型是层的简单堆叠，无法表示任意模型；

   -  使用 Keras 函数式 API 可以构建复杂的模型：

      -  多输入模型

      -  多输出模型

      -  具有共享层的模型(同一层被调用多次)

      -  具有非序列数据流的模型(剩余连接)

   -  使用 Keras 函数式 API 可以：

      -  层实例可调用并返回张量

      -  输入张量和输出张量用于定义 tf.keras.Model 实例

      -  模型的训练方式和 tf.keras.Sequential 模型相同

.. _header-n178:

(1) 使用 Keras 函数式 API 构建一个简单的全连接网络
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # 训练数据
   import numpy as np
   data = np.random.random((1000, 32))
   labels = np.random.random((1000, 10))

   # 用 Keras 函数式 API 构建模型
   inputs = tf.keras.Input(shape = (32,))
   x = layers.Dense(64, activation = "relu")(input)
   x = layers.Dense(64, activation = "relu")(x)
   predictions = layers.Dense(10, activation = "softmax")(x)
   model = tf.keras.Model(inputs = inputs, outputs = predictions)

   # 编译模型
   model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
                 loss = "categorical_corssentropy",
                 metircs = ["accuracy"])

   # 训练模型
   model.fit(data, labels, batch_size = 32, epochs = 5)

.. _header-n180:

3. 模型子类化
-------------

   -  通过对 ``tf.keras.Model``
      进行\ **子类化**\ 并\ **定义自己的前向传播**\ 来构建\ **完全可自定义的模型**\ ；

      -  在 ``__init__``\ 方法中创建层，并将他们设置为类实例的属性

      -  在 ``call`` 方法中定义前向传播；

**使用自定义前向传播进行子类化的 ``tf.keras.Model``\ ：**

.. code:: python

   class MyModel(tf.keras.Model):

       def __init__(self, num_classes = 10):
           super(MyModel, self).__init__(name = "my_model")
           self.num_classes = num_classes

           # 定义自己的层
           self.dense_1 = layers.Dense(32, activation = "relu")
           self.dense_2 = layers.Dense(num_classes, activation = "sigmoid")

       def call(self, inputs):
           # 用在__init__中定义的层，定义自己的前向传播
           x = self.dense_1(inputs)
           return self.dense_2(x)

       def compute_output_shape(self, input_shape):
           shape = tf.TensorShape(input_shape).as_list()
           shape[-1] = self.num_classes
           return tf.TensorShape(shape)


   # 实例化新模型类
   model = MyModel(num_classes = 10)

   # 编译模型
   model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
                 loss = "categorical_crossentropy",
                 metrics = ["accuracy"])

   # 训练模型
   model.fit(data, labels, batch_size = 32, epochs = 5)

.. _header-n192:

4. 自定义层
~~~~~~~~~~~

   -  通过对 ``tf.keras.layers.Layers``
      进行\ **子类化**\ 并实现以下方法来\ **创建自定义层**\ ：

      -  ``build``

         -  创建层的权重；使用 ``add_weight`` 方法添加权重；

      -  ``call``

         -  定义前向传播;

      -  ``compute_output_shape``

         -  指定在给定输入形状的情况下如何计算层的输出形状;

      -  可以通过实现 ``get_config`` 方法和 ``from_config``
         类方法序列化层;

.. code:: python

   class MyLayer(layers.Layer):

       def __init__(self, output_dim, **kwargs):
           self.output_dim = output_dim
           super(MyLayer, self).__init__(**kwargs)

       def build(self, input_shape):
           shape = tf.TensorShape((input_shape[1], self.output_dim))
           self.kernel = self.add_weight(name = "kernel",
                                         shape = shape,
                                         initializer = "uniform",
                                         trainable = True)
           super(MyLayer, self).build(input_shape)

       def call(self, inputs):
           return tf.matmul(inputs, self.kernel)

       def compute_output_shape(self, input_shape):
           shape = tf.TensroShape(input_shape).as_list()
           shape[-1] = self.output_dim
           return tf.TensorShape(shape)

       def get_config(self):
           base_config = super(MyLayer, self).get_config()
           base_config["output_dim"] = self.output_dim
           return base_config

       @classmethod
       def from_config(cls, config):
           return cls(**config)


   model = tf.keras.Sequential([
       MyLayer(10),
       layers.Activation("softmax")])

   model.compile(optimizer = tf.train.RMSPropOPtimizer(0.001),
                 loss = "categorical_crossentropy",
                 metrics = ["accuracy"])

   model.fit(data, labels, batch_size = 32, epochs = 5)

.. _header-n217:

5. 回调
-------

   -  回调是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为；可以编写自定义回调，也可以使用包含以下方法的内置
      ``tf.keras.callbacks``\ ：

      -  ``tf.keras.callbacks.ModelCheckpoint``

         -  定期保存模型的检查点；

      -  ``tf.keras.callbacks.LearningRateScheduler``

         -  动态更改学习率；

      -  ``tf.keras.callbacks.EarlyStopping``

         -  在验证效果不再改进时中断训练；

      -  ``tf.keras.callbacks.TensorBoard``

         -  使用 TensorBoard 监控模型的行为；

   -  要使用 ``tf.keras.callbacks.Callback``\ ，需要将其传递给模型的
      ``fit`` 方法；

.. code:: python

   callbacks = [
       tf.keras.callbacks.ModelCheckpoint(),
       tf.keras.callbacks.LearningRateScheduler(),
       # Interrupt training if `val_loss` stops improving for over 2 epochs
       tf.keras.callbacks.EarlyStopping(patience = 2, monitor = "val_loss"),
       # Write TensorBoard logs to `./logs` directory
       tf.keras.callbacks.TensorBoard(log_dir = "./logs")
   ]

   model.fit(data, labels, 
             batch_size = 32, 
             epochs = 5, 
             callbacks = callbacks, 
             validation_data = (val_data, val_labels))

.. _header-n248:

6. 保存和恢复模型
-----------------

   1. 保存和恢复-仅限权重

      -  ``tf.keras.Model.save_weights``

      -  ``tf.keras.Model.load_weights``

      -  默认情况下，会以 TensorFlow
         检查点文件格式保存模型的权重。权重也可以另存为 Keras HDF5
         格式（Keras 多后端实现的默认格式）；

   2. 保存和恢复-仅限配置

   3. 保存和恢复-整个模型

**保存和恢复仅限权重：**

.. code:: python

   model = tf.keras.Sequential([
       layers.Dense(64, activation = "relu"),
       layers.Dense(10, activation = "softmax")
   ])

   model.compile(optimizer = tf.train.AdamOptimizer(0.001),
                 loss = "categorical_crossentropy",
                 metrics = ["accuracy"])

   # TensorFlow 检查点格式
   model.save_weights("./weights/my_model")
   model.load_weights("./weights/my_model")

   # HDF5 格式
   model.save_weights("my_model.h5", save_format = "h5")
   model.load_weights("my_model.h5")

**保存和恢复仅限配置：**

-  可以保存模型的配置，此操作会对模型架构(不含权重)进行序列化，即使没有定义原始模型的代码，保存的配置也可以重新初始化相同的模型

   -  Keras 支持 JSON 和 YAML 序列化格式

.. code:: python

   # 序列化一个模型为 JSON 格式
   import json
   from pprint import pprint
   json_string = model.to_json()
   ppprint(json.loads(json_string))


   # 从 JSON 重新创建模型(初始化)
   fresh_model = tf.keras.models.model_from_json(json_string)

.. code:: python

   # 序列化一个模型为 YAML 格式
   yaml_string = model.to_yaml()
   print(yaml_string)

   # 从 YAML 重新创建模型(初始化)
   fresh_mddel = tf.keras.models.model_from_yaml(yaml_string)

**保存和恢复整个模型：**

-  整个模型可以保存到一个文件中，其中包含权重值、模型配置、优化器配置；

-  可以对模型设置检查点，并可以从完全相同的状态继续训练，而无需访问原始代码；

.. code:: python

   # 创建一个模型
   model = tf.keras.Sequential([
       layers.Dense(10, activation = "softmax", input_shape(32,)),
       layers.Dense(10, activation = "softmax")
   ])

   # 编译模型
   model.compile(optimizer = "rmsprop",
                 loss = "categorical_crossentropy",
                 metrics = ["accuracy"])

   # 训练模型
   model.fit(data, labels, batch_size = 32, epochs = 5)

   # 保存模型
   model.save("my_model.h5")

   # 重新创建模
   model = tf.keras.models.load_model("my_model.h5")

.. _header-n283:

7. 分布式训练模型
-----------------

-  Estimator API

   -  Estimator API 用于针对分布式环境训练模型

   -  可以将现有的 Keras 模型转换为 Estimator，这样，Keras
      模型就可以利用 Estimator 的优势，比如进行分布式训练

      -  ``tf.keras.Model`` 可以通过 ``tf.estimator`` API
         进行训练，方法是将该模型转换为
         ``tf.estimator.Estimator``\ 对象，通过
         ``tf.keras.estimator.model_to_estimator`` 进行转换；

-  多个 GPU

   -  ``tf.keras`` 模型可以使用
      ``tf.contrib.distribute.DistributionStrategy`` 在多个 GPU
      上运行，这个 API 在多个 GPU
      上提供分布式训练，几乎不需要更改现有代码；

   -  目前，\ ``tf.contrib.distribute.MirroredStrategy``
      是唯一受支持的分布策略。

      -  MirroredStrategy
         通过在一台机器上使用规约在同步训练中进行图内复制。

      -  要将 DistributionStrategy 与 Keras 搭配使用，请将
         tf.keras.Model 转换为 tf.estimator.Estimator（通过
         tf.keras.estimator.model\ *to*\ estimator），然后训练该
         Estimator；

将 Keras 模型(\ ``tf.keras``)转换为 Estimator 进行分布式训练

.. code:: python

   model = tf.keras.Sequential([
       layers.Dense(10, activation = "softmax"),
       layers.Dense(10, activation = "softmax")
   ])
   model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
                 loss = "categorical_crossentropy",
                 metrics = ["accuracy"])
   estimator = tf.keras.estimator.model_to_estimator(model)

多个 GPU 分布式训练模型

创建一个简单的模型：

.. code:: python

   model = tf.keras.Sequential([
       layers.Dense(16, activation = "relu", input_shape = (10)),
       layers.Dense(1, activation = "sigmoid")
   ])
   model.compile(optimizer = tf.train.GradientDescentOptimizer(0.2),
                 loss = "binary_crossentropy",
                 metrics = ["accuracy"])
   model.summary()

定义输入管道：

.. code:: python

   def input_fn():
       x = np.random.random((1024, 10)),
       y = np.random.randint(2, size = (1024, 1))
       x = tf.cast(x, tf.float32)
       dataset = tf.data.Dataset.from_tensor_slices((x, y))
       dataset = dataset.repeat(10)
       dataset = dataset.batch(32)
       return dataset

创建分布式配置：

.. code:: python

   strategy = tf.contrib.distribute.MirroredStrategy(
           # 设备list
           # num_gpus
       )
   config = tf.estimator.RunConfig(train_distribute = strategy)

将 Keras 模型转换为 ``tf.estimator.Estimator`` 实例：

.. code:: python

   keras_estimator = tf.keras.estimator.model_to_estimator(
       keras_model = model,
       config = config,
       model_dir = "/tmp/model_dir"
   )

通过提供 ``input_fn`` 和 ``steps`` 参数训练 ``Estimator`` 实例：

.. code:: python

   keras_estimator.train(input_fn = input_fn, steps = 10)

.. _header-n322:

8. TensorFlow Keras API
-----------------------

.. code:: python

   import tensorflow as tf

..

   Keras API 架构：

   -  Module: tf.keras

      -  Modules

         -  models

         -  layers

         -  optimizers

         -  losses

            -  Keras 内置损失函数

         -  metrics

         -  tf.keras.activations

            -  Keras 内置激活函数

         -  applications

         -  backend

         -  callbacks

         -  constraints

         -  datasets

         -  estimator

         -  expreimental

         -  initializers

         -  preprocessing

         -  regularizers

         -  utils

         -  wrappers

      -  Classes

         -  ``class`` tf.keras.Sequential()

         -  ``class`` tf.keras.Model()

      -  Functions

         -  Input(...)

            -  用来实例化一个 Keras tensor

      -  Other Members

         -  ``__version__``

模型

层

-  CNN

   -  卷积层

      -  tf.keras.layers.Conv1D

      -  tf.keras.layers.Conv2D

      -  tf.keras.layers.Conv2DTranspose

      -  tf.keras.layers.Conv3D

      -  tf.keras.layers.Conv3DTranspose

      -  tf.keras.layers.ConvLSTM2D

   -  池化层

      -  tf.keras.layers.MaxPool1D

      -  tf.keras.layers.MaxPool2D

      -  tf.keras.layers.MaxPool3D

   -  全连接层

      -  tf.keras.layers.Dense

-  DNN

   -  

-  正则化

   -  

-  激活函数

   -  tf.kearas.layers.ReLU

   -  tf.keras.layers.Softmax

激活函数

-  tf.keras.activations.get(indentifier)

-  tf.keras.activations.serialize(activation)

   -  序列化

-  tf.keras.activations.deserialize(name, custom_objects = None)

   -  反序列化

-  tf.keras.activations.linear(x)

-  tf.keras.activations.sigmoid(x)

-  tf.keras.activations.hard_sigmoid(x)

-  tf.keras.activations.relu(x, alpha = 0.0, max_value = None, threshold
   = 0)

-  tf.keras.activations.softmax(x, axis = -1)

-  tf.keras.activations.tanh(x)

   -  :math:`h(x) = tanh(x)`

-  tf.keras.activations.softplus(x)

   -  :math:`h(x) = log(e^{x} + 1)`

-  tf.keras.activations.softsign(x)

   -  :math:`h(x) = \frac{x}{|x| + 1}`

-  tf.keras.activations.exponential(x)

   -  指数

-  tf.keras.activations.elu(x, alpha = 1.0)

   -  指数线性单元

-  tf.keras.activations.selu(x)

   -  标准化的指数线性单元

梯度下降算法

Class:

-  tf.keras.optimizers.Adadelta()

-  tf.keras.optimizers.Adagrad()

-  tf.keras.optimizers.Adam()

-  tf.keras.optimizers.Nadam()

-  tf.keras.optimizers.Optimizer()

-  tf.keras.optimizers.RMSprop()

-  tf.keras.optimizers.SGD()

   -  SGD

      -  tf.keras.optimizers.SGD(lr = 0.01)

   -  Momentum SGD

      -  tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9)

   -  Nesterov momentum SGD

      -  tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9, nesterov =
         True)

   -  learning rate decay SGD

      -  tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.0)

Function:

-  序列化函数

   -  serialize()

-  反序列化函数

   -  deserialize()

-  检索优化器实例

   -  get()

损失函数, 评估准则

Classes:

-  tf.keras.losses

   -  分类

      -  二元交叉熵损失

         -  ``tf.keras.losses.BinaryCrossentropy()``

            -  'binary_crossentropy'

      -  类别交叉熵损失

         -  ``tf.keras.losses.CategoricalCrossentropy()``

            -  'categorical_crossentropy'

   -  回归

      -  绝对平均误差损失

         -  ``tf.keras.losses.MeanAbsoluteError()``

            -  ''

      -  绝对平均百分比误差损失

         -  ``tf.keras.losses.MeanAbsolutePercentageError()``

            -  ''

      -  均方误差损失

         -  ``tf.keras.losses.MeanSquaredError``

      -  均方对数误差损失

         -  ``tf.keras.losses.MeanSquaredLogarithmicError()``

         -  ''

-  tf.keras.metrics

   -  分类

      -  ``tf.keras.metrics.Accuracy()``

      -  ``tf.keras.metrics.BinaryAccuracy()``

      -  ``tf.keras.metrics.CategoricalAccuracy()``

      -  ``tf.keras.metrics.FalseNegatives()``

      -  ``tf.keras.metrics.FalsePositives()``

      -  ``tf.keras.metrics.Precision()``

      -  ``tf.keras.metrics.Recall()``

      -  ``tf.keras.metrics.SensitivityAtSpecificity()``

      -  ``tf.keras.metrics.SparseCategoricalAccuracy()``

      -  ``tf.keras.metrics.SpecificityAtSensitivity()``

      -  ``tf.keras.metrics.TrueNegatives()``

      -  ``tf.keras.metrics.TruePositives()``

   -  回归

      -  ``tf.keras.metrics.Mean()``

Functions:

-  tf.keras.[``losses``].categorical_hinge(...)

-  tf.keras.[``losses``].logcosh(...)

   -  Logarithm of the hyperbolic cosine of the prediction error.

-  tf.keras.[``metrics``].categorical_accuracy(...)

-  tf.keras.[``metrics``].binary_accuracy(...)

-  tf.keras.[``metrics``].sparse\ *categorical*\ accuracy(...)

-  tf.keras.[``metrics``].sparse\ *top*\ k\ *categorical*\ accuracy(...)

-  tf.keras.[``metrics``].top\ *k*\ categorical_accuracy(...)

-  tf.keras.[\ ``metrics``/``loss``].KLD(...)

-  tf.keras.[\ ``metrics``/``loss``].MAE(...)

-  tf.keras.[\ ``metrics``/``loss``].MAPE(...)

-  tf.keras.[\ ``metrics``/``loss``].MSE(...)

-  tf.keras.[\ ``metrics``/``loss``].MSLE(...)

-  tf.keras.[\ ``metrics``/``loss``].binary_crossentropy(...)

-  tf.keras.[\ ``metrics``/``loss``].categorical_crossentropy(...)

-  tf.keras.[\ ``metrics``/``loss``].cosine(...)

-  tf.keras.[\ ``metrics``/``loss``].cosine_proximity(...)

-  tf.keras.[\ ``metrics``/``loss``].deserialize(...)

-  tf.keras.[\ ``metrics``/``loss``].get(...)

-  tf.keras.[\ ``metrics``/``loss``].hinge(...)

-  tf.keras.[\ ``metrics``/``loss``].kld(...)

-  tf.keras.[\ ``metrics``/``loss``].kullback\ *leibler*\ divergence(...)

-  tf.keras.[\ ``metrics``/``loss``].mae(...)

-  tf.keras.[\ ``metrics``/``loss``].mape(...)

-  tf.keras.[\ ``metrics``/``loss``].mean\ *absolute*\ error(...)

-  tf.keras.[\ ``metrics``/``loss``].mean\ *absolute*\ percentage_error(...)

-  tf.keras.[\ ``metrics``/``loss``].mean\ *squared*\ error(...)

-  tf.keras.[\ ``metrics``/``loss``].mean\ *squared*\ logarithmic_error(...)

-  tf.keras.[\ ``metrics``/``loss``].mse(...)

-  tf.keras.[\ ``metrics``/``loss``].msle(...)

-  tf.keras.[\ ``metrics``/``loss``].poisson(...)

-  tf.keras.[\ ``metrics``/``loss``].serialize(...)

-  tf.keras.[\ ``metrics``/``loss``].sparse\ *categorical*\ crossentropy(...)

-  tf.keras.[\ ``metrics``/``loss``].squared_hinge(...)

正则化

参数初始化
