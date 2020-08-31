
Keras-MNIST-RNN-Class
=====================

1. Keras Sequential Model
-------------------------

导入模块
~~~~~~~~

.. code:: python

   import numpy as np
   np.random.seed(1337)
   from keras.datasets import mnist
   from keras import utils
   from keras.models import Sequential
   from keras.layers import SimpleRNN, Activation, Dense
   from keras.optimizers import Adam


数据预处理
~~~~~~~~~~

.. code:: python

   TIME_STEPS = 28
   INPUT_SIZE = 28
   BATCH_SIZE = 50
   BATCH_INDEX = 0
   OUTPUT_SIZE = 10
   CELL_SIZE = 50
   LR = 0.001

.. code:: python

   # load data
   (X_train, y_train), (X_test, y_test) = mnist.load_data()

   # data preprocessing
   X_train = X_train.reshape(-1, 1, 28, 28) / 255.
   X_test = X_test.reshape(-1, 1, 28, 28) / 255.
   y_train = utils.to_categorical(y_train, num_classes = 10)
   y_test = utils.to_categorical(y_test, num_classes = 10)


建立模型
~~~~~~~~

.. code:: python

   # RNN model
   model = Sequential()
   # RNN cell
   model.add(SimpleRNN(
      batch_input_shape = (None, TIME_STEPS, INPUT_SIZE),
      output_dim = CELL_SIZE,
      unroll = True
   ))
   # output layer
   model.add(Dense(OUTPUT_SIZE))
   model.add(Activation("softmax"))


模型编译
~~~~~~~~

.. code:: python

   adam = Adam(LR)
   model.compile(optimizer = adam,
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])


模型训练
~~~~~~~~

.. code:: python

   for step in range(4001):
      X_batch = X_train[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :, :]
      Y_batch = y_train[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :]
      cost = model.train_on_batch(X_batch, Y_batch)
      BATCH_INDEX += BATCH_SIZE
      BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

      if setp % 500 == 0:
         cost, accuracy = model.evaluate(X_test, y_test, batch_size = y_test.shape[0], verbose = False)
         print("test cost: ", cost)
         print("test accuracy: ", accuracy)

模型评估
~~~~~~~~

.. code:: python

   test


模型预测
~~~~~~~~

.. code:: python

   test

模型结果输出
~~~~~~~~~~~~

.. code:: python

   test

2. Keras 函数式 API Model
-------------------------
