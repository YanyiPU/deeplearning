.. _header-n0:

Keras-Linear-Regression
=======================

.. _header-n3:

导入模块
--------

.. code:: python

   import numpy as np
   np.random.seed(1337)
   import matplotlib.pyplot as plt
   from keras import layers
   from keras.models import Sequential, Model

.. _header-n5:

创建数据
--------

.. code:: python

   # create some data
   X = np.linspace(-1, 1, 200)
   np.random.shuffle(X)
   Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

   # plot data
   plt.scatter(X, Y)
   plt.show()

   X_train, Y_train = X[:160], Y[:160]
   X_test, Y_test = X[160:], Y[160:]

.. _header-n7:

建立模型
--------

.. code:: python

   model = Sequential()
   model.add(Dense(output_dim = 1, input_dim = 1))

.. _header-n9:

编译模型
--------

.. code:: python

   model.compile(loss = "mse",
   			  optimizer = "sgd")

.. _header-n11:

训练模型
--------

.. code:: python

   # training 
   print("Training -------------")
   for step in range(301):
   	cost = model.train_on_batch(X_train, Y_train)
   	if step % 100 == 0:
   		print("train cost: ", cost)

.. _header-n13:

验证模型
--------

.. code:: python

   # test 
   print("\nTesting -------------")
   cost = model.evaluate(X_test, Y_test, batch_size = 40)
   print("test cost: ", cost)
   W, b = model.layers[0].get_weight()
   print("Weights = ", W)
   print("Biases = ", b)

.. _header-n15:

模型预测及模型结果可视化
------------------------

.. code:: python

   # plotting the prediction
   Y_pred = model.predict(X_test)
   plt.scatter(X_test, Y_test)
   plt.plot(X_test, Y_test)
   plt.show()
