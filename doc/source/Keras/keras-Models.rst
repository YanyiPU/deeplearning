.. _header-n0:

Keras 模型
==========

-  Sequential model

-  Model class used with the function API

.. _header-n8:

1.Keras 模型共有的方法和属性
----------------------------

.. code:: python

   from keras.model import Model
   from keras.model import model_from_json, model_from_yaml

-  model.layers

-  model.inputs

-  model.outputs

-  model.summary()

-  Config

   -  model.get_config()

      -  Model.from_config()

      -  Sequential.from_config()

-  Weights

   -  model.get_weights()

      -  *to Numpy arrays*

   -  model.set_weights(weights)

      -  *from Numpy arrays*

   -  model.save_weights(filepath)

      -  *to HDF5 file*

   -  model.load\ *weights(filepath, by*\ name = False)

      -  *from HDF5 file*

-  Save or Load

   -  model.to_json()

      -  model\ *from*\ json()

   -  model\ *to*\ yaml()

      -  model\ *from*\ yaml()

.. _header-n65:

2.Model subclassing
-------------------

   -  构建 full-customizable model by subclassing the ``Model`` class

   -  实现 forward pass in the ``call`` method

-  模型的 layers 定义在 ``__init__(self, ...)`` 中

-  模型的前向传播定义在 ``call(self, inputs)`` 中

-  可以通过调用制定的自定义损失函数 ``self.add_loss(loss_tensor)``

-  在 subclassing 模型中，模型的拓扑结构被定义为 Python 代码，而不是
   layers
   的静态图，因此无法检查或序列化模型的拓扑结构，即以下方法不适用于
   subclassing 模型：

   -  model.inputs

   -  model.outputs

   -  model.to_yaml()

   -  model.to_json()

   -  model.get_config()

   -  model.save()

-  模型(keras.model.Model)子类的 API
   可以为实现更加复杂的模型提供了灵活性，但是是有代价的，除了以上的功能不能使用，并且模型更复杂，更容易出错

示例：

.. code:: python

   import keras

   class SimpleMLP(keras.Model):

       def __init__(self, use_bn = False, use_dp = False, num_classes = 10):
           super(SimpleMLP, self).__init__(name = "mlp")
           self.use_bn = use_bn
           self.use_dp = use_dp
           self.num_classes = num_classes

           # layers    
           self.dense1 = keras.layers.Dense(32, activation = "relu")
           self.dense2 = keras.layers.Dense(num_classes, activation = "softmax")
           if self.use_dp:
               self.dp = keras.layers.Dropout(0.5)
           if self.use_bn:
               self.bn = keras.layers.BatchNormalization(axis = -1)

       def call(self, inputs):
           """
           前向传播
           """
           x = self.dense1(inputs)
           if self.use_dp:
               x = self.dp(x)
           if self.use_bn:
               x = self.bn(x)

           return self.dense2(x)

   model = SimpleMLP()
   model.compile(...)
   model.fit(...)

.. _header-n100:

3.Keras Sequential 模型的使用文档
---------------------------------

   Sequential 模型是层(layers)的线性堆叠

**3.1 Keras Sequential Hello World**

.. code:: python

   # in Python
   from keras.model import Sequential
   from keras.layers import Dense, Activation

   # ==========
   # 模型构建
   # ==========
   model = Sequential()
   model.add(Dense(units = 64, activation = "relu", input_dim = 784))
   model.add(Dense(units = 64, activation = "relu"))
   model.add(Dense(units = 10, activation = "softmax"))

   # model = Sequential([
   #     Dense(64, input_shape = (784,)),
   #     Activation("relu"),
   #     Dense(64),
   #     Activation("relu")
   #     Dense(10),
   #     Activation("softmax")
   # ])

   # ==========
   # 模型编译
   # ==========
   # model.compile(loss = "categorical_crossentropy",
   #               optimizer = "sgd",
   #               metrics = ["accuracy"])

   model.compile(loss = keras.losses.categorical_crossentropy
                 optimizer = keras.optimizer.SGD(lr = 0.01, momentum = 0.9, nesterov = True),
                 metrics = keras.metircs.Accuracy)

   # ==========
   # 模型训练
   # ==========
   model.fit(x_train, y_train, epochs = 5, batch_size = 32)
   # model.train_on_batch(x_batch, y_batch)

   # ==========
   # 模型评估
   # ==========
   loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)

   # ==========
   # 模型预测
   # ==========
   classes = model.predict(x_test, batch_size = 128)

.. _header-n106:

4.Keras 函数式API 的使用文档
----------------------------

   -  Keras 函数式 API 是定义复杂模型的方法

   -  Keras 函数式 API
      可以重用经过训练的模型，可以通过在张量上调用任何模型并将其视为一个层(layers)

      -  调用模型的结构

      -  调用模型的权重

**4.1 Keras 函数式API Hello World**

**A densely-connected network:**

.. code:: python

   from keras.layers import Input, Dense
   from keras.models import Model

   # ==========
   # 模型构建
   # ==========
   inputs = Input(shape = (784,))
   x = Dense(64, activation = "relu")(inputs)
   x = Dense(64, activation = "relu")(x)
   predictions = Dense(10, activation = "softmax")(x)
   model = Model(inputs = inputs, outputs = predictions)

   # ==========
   # 模型编译
   # ==========
   model.compile(optimizer = "rmsprop",
                 loss = "categorical_crossentropy",
                 metrics = ["accuracy"])

   # ==========
   # 模型训练
   # ==========
   model.fit(data, labels)

**4.2 函数式 API 特点**

   -  所有模型都像层(layer)一样可以调用

   -  多输入和多输出模型

   -  共享图层

   -  "层节点"概念

**所有模型都像层(layer)一样可以调用：**

.. code:: python

   # 将图像分类模型转换为视屏分类模型
   from keras.layers import TimeDistributed
   from keras.layers import Input, Dense
   from keras.models import Model

   inputs = Input(shape = (784,))
   x = Dense(64, activation = "relu")(inputs)
   x = Dense(64, activation = "relu")(x)
   predictions = Dense(10, activation = "softmax")(x)
   model = Model(inputs = inputs, outputs = predictions)

   input_sequences = Input(shape = (20, 784))
   processed_sequences = TimeDistributed(model)(input_sequences)

**多输入和多输出模型：**

.. code:: python

   from keras.layers import Input, Embedding, LSTM, Dense
   from keras.models import Model

   # ==========
   # 模型构建
   # ==========
   # 标题输入
   main_input = Input(shape = (100,), dtype = "int32", name = "main_input")
   x = Embedding(output_dim = 512, input_dim = 10000, input_length = 100)(main_input)
   lstm_out = LSTM(32)(x)
   auxiliary_output = Dense(1, activation = "sigmoid", name = "aux_output")(lstm_out)

   # 标题发布时间等数据输入
   auxiliary_input = Input(shape = (5,), name = "aux_input")

   # concatenate the lstm_out and auxiliary_input
   x = keras.layers.concatenate([lstm_out, auxiliary_input])
   x = Dense(64, activation = "relu")(x)
   x = Dense(64, activation = "relu")(x)
   x = Dense(64, activation = "relu")(x)
   main_output = Dense(1, activation = "sigmoid", name = "main_output")(x)
   model = Model(inputs = [main_input, auxiliary_input], 
                 outputs = [main_output, auxiliary_output])

   # ==========
   # 模型编译
   # ==========
   model.compile(optimizer = "rmsprop",
                 loss = {
                   "main_output": "binary_crossentropy", 
                   "aux_output": "binary_crossentropy"
                 },
                 loss_weights = {
                   "main_output": 1,
                   "aux_output": 0.2
                 })

   # ==========
   # 模型训练
   # ==========
   model.fit(
       {
           "main_input": headline_data,
           "aux_input": additional_data
       },
       {
           "main_output": labels,
           "aux_output": labels
       },
       epochs = 50,
       batch_size = 32
   )

**共享图层：**

任务：判断两条推文是否来自同一个人

.. code:: python

   import keras
   from keras.layers import Input, LSTM, Dense
   from keras.models import Model

   # ==========
   # 数据处理
   # ==========


   # ==========
   # 模型构建
   # ==========
   # input layers
   tweet_a = Input(shape = (280, 256))
   tweet_b = Input(shape = (280, 256))
   # LSTM layers
   shared_lstm = LSTM(64)
   encoded_a = shared_lstm(tweet_a)
   encoded_b = shared_lstm(tweet_b)
   # concatenate two vectors
   merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis = -1)
   # output layers(add a logistic regression on top)
   predictions = Dense(1, activation = "sigmoid")(merged_vector)
   model = Model(inputs = [tweet_a, tweet_b],
                 output = predictions)

   # ==========
   # 模型编译
   # ==========
   model.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["accuracy"])

   # ==========
   # 模型训练
   # ==========
   model.fit([data_a, data_b], epochs = 10)

**"层节点"概念：**

输出层连接到单个输入层：

.. code:: python

   from keras.layers import Input, LSTM

   a = Input(shape = (280, 256))

   lstm = LSTM(32)
   encoded_a = lstm(a)

   # assert lstm.output == encoded_a

   output = lstm.output
   output_shape = lstm.output_shape

输出层连接到多个输入层：

.. code:: python

   from keras.layers import Input, LSTM
   a = Input(shape = (280, 256))
   b = Input(shape = (280, 256))

   lstm = LSTM(32)

   encoded_a = lstm(a)
   encoded_b = lstm(b)

   # lstm.output
   # lstm.output_shape
   # assert lstm.get_output_at(0) == encoded_a
   # assert lstm.get_output_at(1) == encoded_b

   output0 = lstm.get_output_at(0)
   output1 = lstm.get_output_at(1)
   output0_shape = lstm.get_output_shape_at(0)
   output1_shape = lstm.get_output_shape_at(1)

**4.3 函数式 API 模型例子**

Inception module

`论文 <https://arxiv.org/pdf/1409.4842.pdf>`__

.. code:: python

   from keras.layers import Input, Conv2D, MaxPooling2D
   from keras.models import Model

   input_img = Input(shape = (256, 256, 3)) 
   tower_1 = Conv2D(64, (1, 1), padding = "same", activation = "relu")(input_img)
   tower_1 = Conv2D(64, (3, 3), padding = "same", activation = "relu")(tower_1)

   tower_2 = Conv2D(64, (1, 1), padding = "same", activation = "relu")(input_img)
   tower_2 = Conv2D(64, (5, 5), padding = "same", activation = "relu")(tower_2)

   tower_3 = MaxPooling2D((3, 3), strides = (1, 1), padding = "same")(input_img)
   tower_3 = Conv2D(64, (1, 1), padding = "same", activation = "relu")(tower_3)

   output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 1)

   model = Modle(input = "",
                 output = "")

   model.compile()

   model.fit()

Residual connection on a convolution layer

`论文 <https://arxiv.org/pdf/1512.03385.pdf>`__

.. code:: python

   from keras.layers import Input, Conv2D
   from keras.model import Model

   x = Input(shape = (256, 256, 3))
   y = Conv2D(3, (3, 3), padding = "same")(x)
   output = keras.layers.add([x, y])

   model = Model(input = "",
                 output = "")

   model.compile()

   model.fit()

Shared vision model Visual question answering model Video question
answering model

.. _header-n154:

5.回调函数-Callbacks
--------------------

-  回调函数是一个函数的集合，会在训练的阶段使用

-  可以使用回调函数查看训练模型的内在状态和统计。也可以传递一个列表的回调函数(作为
   ``callbacks``\ 关键字参数)到 ``Sequential`` 或 ``Model`` 类型的
   ``.fit()`` 方法。在训练时，相应的回调函数的方法会被在各自的阶段被调用

回调函数：

-  keras.callbacks.Callback()

   -  用来创建新的回调函数的抽象基类

   -  ``.params``

   -  ``.model``

-  keras.callbacks.BaseLogger(stateful_metrics = None)

   -  基类训练 epoch 评估值的均值

-  keras.callbacks.TerminateOnNaN()

   -  当遇到损失为 ``NaN`` 停止训练

-  keras.callbacks.ProgbarLogger()

-  keras.callbacks.History()

   -  所有事件都记录到 History 对象

-  keras.callbacks.ModelCheckpoint()

   -  在每个训练期之后保存模型

-  keras.callbacks.EarlyStopping()

-  keras.callbacks.RemoteMonitor()

-  keras.callbacks.LearningRateScheduler(schedule, verbose = 0)

-  keras.callbacks.TensorBoard()

-  keras.callbacks.ReduceLROnPlateau()

-  keras.callbacks.CSVLogger()

-  keras.callbacks.LambdaCallback()

**创建回调函数:**

.. code:: python

   from keras.layers import Dense, Activation
   from keras.models import Sequential
   from keras.callbacks import ModelCheckpoint

   # 模型建立
   model = Sequenital()
   model.add(Dense(10, input_dim = 784, kernel_initializer = "uniform"))
   model.add(Activation("softmax"))

   # 模型编译
   model.compile(loss = "categorical_crossentropy",
                 optimizer = "rmsporp")

   # 模型训练
   # 在训练时，保存批量损失值
   class LossHistory(keras.callbacks.Callback):
       def on_train_begin(self, logs = {}):
           self.losses = []

       def on_batch_end(self, batch, logs = {}):
           self.losses.append(logs.get("loss"))
   history = LossHistory()

   # 如果验证集损失下降，在每个训练 epoch 后保存模型
   checkpointer = ModelCheckpoint(filepath = "/tmp/weight.hdf5",
                                  verbose = 1,
                                  save_best_only = True)
   model.fit(x_train, y_train, 
             batch_size = 128, epochs = 20, 
             verbose = 0,
             validation_data = (x_test, y_test), 
             callbacks = [history, checkpointer])

   # 模型结果输出
   print(history.losses)

.. _header-n210:

6.Applications
--------------

Keras Applications(\ ``keras.applications``)
提供了预训练好的深度学习模型，这些模型可以用于预测、特征提取等.

当初始化一个模型时就会自动下载，默认下载的路径是：\ ``~/.keras.models/``.

**可用的模型：**

   在 ImageNet 数据上预训练过的用于图像分类的模型

-  Xception

-  VGG16

-  VGG19

-  ResNet, ResNetV2, ResNeXt

-  InceptionV3

-  InceptionResNet2

-  MobileNet

-  MobileNetV2

-  DenseNet

-  NASNet

.. code:: python

   from keras.applications.xception import Xception
   from keras.applications.vgg16 import VGG16
   from keras.applications.vgg19 import VGG19
   from keras.applications.resnet50 import ResNet50
   from keras.applications.inception_v3 import InceptionV3
   from keras.applications.inception_resnet_v2 import InceptionResNetV2
   from keras.applications.mobilenet import MobileNet
   from keras.applications.densenet import DenseNet121
   from keras.applications.densenet import DenseNet169
   from keras.applications.densenet import DenseNet201
   from keras.applications.nasnet import NASNetLarge
   from keras.applications.nasnet import NASNetMobile
   from keras.applications.mobilenet_v2 import MobileNetV2

   # channels_last only; 299x299
   xception_model = Xception(include_top = True,
                             weights = "imagenet",
                             input_tensor = None, 
                             input_shape = None,
                             pooling = None,
                             classes = 1000)
   # channels_first and channels_last; 224x224
   vgg16_model = VGG16(include_top = True,
                       weights = "imagenet",
                       input_tensor = None, 
                       input_shape = None,
                       pooling = None,
                       classes = 1000)
   vgg19_model = VGG19(include_top = True, 
                       weights = 'imagenet',
                       input_tensor = None, 
                       input_shape = None, 
                       pooling = None, 
                       classes = 1000)
   resnet50_model = ResNet50(include_top = True, 
                             weights = 'imagenet', 
                             input_tensor = None, 
                             input_shape = None, 
                             pooling = None, 
                             classes = 1000)
   inception_v3_model = InceptionV3(include_top = True, 
                                    weights = 'imagenet', 
                                    input_tensor = None, 
                                    input_shape = None, 
                                    pooling = None, 
                                    classes = 1000)
   inception_resnet_v2_model = InceptionResNetV2(include_top = True, 
                                                 weights = 'imagenet', 
                                                 input_tensor = None, 
                                                 input_shape = None, 
                                                 pooling = None, 
                                                 classes = 1000)
   mobilenet_model = MobileNet(input_shape = None, 
                               alpha = 1.0, 
                               depth_multiplier = 1, 
                               dropout = 1e-3, 
                               include_top = True, 
                               weights = 'imagenet', 
                               input_tensor = None, 
                               pooling = None, 
                               classes = 1000)
   densenet_model = DenseNet121(include_top = True, 
                                weights = 'imagenet', 
                                input_tensor = None, 
                                input_shape = None, 
                                pooling = None, 
                                classes = 1000)
   densenet_model = DenseNet169(include_top = True, 
                                weights = 'imagenet', 
                                input_tensor = None, 
                                input_shape = None, 
                                pooling = None, 
                                classes = 1000)
   densenet_model = DenseNet201(include_top = True, 
                                weights = 'imagenet', 
                                input_tensor = None, 
                                input_shape = None, 
                                pooling = None, 
                                classes = 1000)
   nasnet_model = NASNetLarge(input_shape = None, 
                              include_top = True, 
                              weights = 'imagenet', 
                              input_tensor = None, 
                              pooling = None, 
                              classes = 1000)
   nasnet_model = NASNetMobile(input_shape = None, 
                               include_top = True, 
                               weights = 'imagenet', 
                               input_tensor = None, 
                               pooling = None, 
                               classes = 1000)
   mobilenet_v2_model = MobileNetV2(input_shape = None, 
                                    alpha = 1.0, 
                                    depth_multiplier = 1, 
                                    include_top = True, 
                                    weights = 'imagenet', 
                                    input_tensor = None, 
                                    pooling = None, 
                                    classes = 1000)

**图像分类模型使用示例：**

.. code:: python

   from keras.preprocessing import image
   from keras.applications.resnet50 import ResNet50
   from keras.applications.resnet50 import preprocess_input, decode_prediction
   import numpy as np

   # Load model
   model = ResNet50(weights = "imagenet")

   # Image data
   img_path = "elephant.jpg"
   img = image.load_img(img_path, target_size = (224, 224))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis = 0)
   x = preprocess_input(x)

   preds = model.predict(x)
   print("Predicted:", decode_prediction(preds, top = 3)[0])
