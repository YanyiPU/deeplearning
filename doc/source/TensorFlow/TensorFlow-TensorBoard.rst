.. _header-n0:

TensorFlow TensorBoard
==========================

TensorFlow 提供了一个名为 TensorBorad 的使用程序。功能是将计算图可视化.

-  Tracking and visualizing metrics such as loss and accuracy

-  Visualizing the model graph (ops and layers)

-  Viewing histograms of weights, biases, or other tensors as they
   change over time

-  Projecting embeddings to a lower dimensional space

-  Dispalying images, text, and audio data

-  Profiling TensorFlow programs

-  And much more

.. _header-n19:

1.开始使用 TensorBoard
----------------------

.. _header-n21:

1.1 载入 TensorBoard notebook extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Load the TensorBoard notebook extension
   %load_ext tensorboard

.. code:: shell

   load_ext tensorboard

.. _header-n24:

1.2 建立模型
~~~~~~~~~~~~

.. code:: python

   import tensorflow as tf 
   import datetime

   # Clear any logs from previous runs
   # $ !rm -rf ./logs/

   # Using the MNIST datasets as the example, normalize the data and write a function that creates a simple Keras models for classifying the images into 10 classes.
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0

   def create_model():
       model = tf.keras.models.Sequential([
           tf.keras.layers.Flatten(input_shape = (28, 28)),
           tf.keras.layers.Dense(512, activation = "relu"),
           tf.keras.layers.Dropout(0.2),
           tf.keras.layers.Dense(10, activation = "softmax")
       ])

       return model

.. _header-n26:

1.3 Using TensorBoard with Keras Model.fit()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   model = create_model()
   model.compile(optimizer = "adam",
                 loss = "sparse_categorical_crossentropy",
                 metrics = ["accuracy"])

   log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1) # enable histogram computation every epoch with `histogram_freq = 1`
   model.fit(x = x_train, 
             y = y_train, 
             epochs = 5, 
             validation_data = (x_test, y_test),
             callbacks = [tensorboard_callback])

.. _header-n29:

1.4 启动 TensorBoard：
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # jupyter notebook
   %tensorboard --logdir logs/fit

.. code:: shell

   # command line
   tensorboard --logdir logs/fit

Dashboards:

-  The **Scalars** dashboard shows how the loss and metrics change with
   every epoch. You can use it to also track training speed, learning
   rate, and other scalar values.

-  The **Graphs** dashboard helps you visualize your model. In this
   case, the Keras graph of layers is shown which can help you ensure it
   is built correctly.

-  The **Distributions** and **Histograms** dashboards show the
   distribution of a Tensor over time. This can be useful to visualize
   weights and biases and verify that they are changing in an expected
   way.

.. _header-n40:

1.5 使用 TensorBoard 的其他方式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   load_ext tensorboard
   rm -rf ./logs/

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-


   import tensorflow as tf
   import datetime


   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0

   train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
   train_dataset = train_dataset.shuffle(60000).batch(64)
   test_dataset = test_dataset.batch(64)

.. code:: python

   loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
   optimizer = tf.keras.optimizer.Adam()

   # Define our metrics
   train_loss = tf.keras.metrics.Mean("train_loss", dtype = tf.float32)
   train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy("train_accuracy")
   test_loss = tf.keras.metrics.Mean("test_loss", dtype = tf.float32)
   test_accuracy = tf.keras.metrics.SparseCatetoricalAccuracy("test_accuracy")

.. code:: python

   def train_step(model, optimizer, x_train, y_train):
       with tf.GradientTape() as tape:
           predictions = model(x_train, training = True)
           loss = loss_object(y_train, predictions)
       grads = tape.gradient(loss, model.trainable_variables)
       optimizer.apply_gradients(zip(grads, model.trainable_variables))

       train_loss(loss)
       train_accuracy(y_train, predictions)


   def test_step(model, x_test, y_test):
       predictions = model(x_test)
       loss = loss_object(y_test, predictions)

       test_loss(loss)
       test_accuracy(y_test, predictions)

.. code:: python

   current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   train_log_dir = "logs/gradient_tape/" + current_time + "/train"
   test_log_dir = "logs/gradient_tape/" + current_time + "/test"
   train_summary_writer = tf.summary.create_file_writer(train_log_dir)
   test_summary_writer = tf.summary.create_file_writer(test_log_dir)

.. _header-n50:

标量和指标
----------

.. _header-n52:

图片数据
--------

.. _header-n54:

模型图
------

.. _header-n56:

超参数调节
----------

.. _header-n58:

Embedding Projector
-------------------

.. _header-n61:

What-If 工具
------------

.. _header-n63:

公平性指标
----------

.. _header-n66:

剖析工具
--------

.. _header-n69:

笔记本中的 TensorBoard
----------------------

.. _header-n72:

TensorBoard.dev 
----------------

.. _header-n74:

(1) Prepare your TensorBoard logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _header-n75:

(2) Upload the logs
~~~~~~~~~~~~~~~~~~~

Install the latest version of TensorBoard to use the uploader.

.. code:: shell

   # Install TensorBoard
   pip install -U tensorboard

   # TensorBoard help
   tensorboard dev --help
   tensorboard dev COMMAND --help

   # Upload
   tensorboard dev upload --logdir logs \
       --name "(optional) My latest experiment" \
       --description "(optional) Simple comparison of several hyperparameters"

输出：

.. code:: 

   ***** TensorBoard Uploader *****

   This TensorBoard will be visible to everyone. Do not upload sensitive data.

   Continue? (yes/NO)

   Please visit this URL to authorize this application: https://accounts.google.com/o/oauth2/auth...

   Uploading to TensorBoard.dev at https://tensorboard.dev/experiment/9E4U9wixQTyeZwGdcehMeA

.. _header-n82:

(3) View your experiment on TensorBoard.dev
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**将计算图保存为 TensorBoard 摘要文件：**

.. code:: python

   import tensorflow as tf

   write = tf.summary.FileWrite('.')
   tb_graph = tf.get_default_graph()
   write.add_graph(tb_graph)

**启动TensorBoard：**

.. code:: shell

   $ tensorboard --logdir .



2.实时查看参数变化情况
------------------------------------









3.查看 Graph 和 Profile 信息
-------------------------------------



