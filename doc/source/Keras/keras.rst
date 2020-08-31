
Keras
=====

1.Keras 介绍
------------

   -  Keras: The Python Deep Learning library

      - ``keras``

      -  ``tensorflow.keras``

   - 为什么要使用 Keras？

      -  Keras 优先考虑开发人员

      -  Keras 已在业界和研究界广泛使用

      -  Keras 使得将模型转化为产品变得容易

      -  Keras 支持多种后端引擎

      -  Keras 具有强大的多 GPU 支持和分布式训练支持

      -  Keras 开发得到了深度学习生态系统中主要公司的支持

2.Keras 入门
--------------

   - Keras 核心数据结构：

      -  ``tensorflow.keras.layers``

      -  ``tensorflow.keras.models``

   - Keras Model 类型：

      -  **Sequential model**

      -  **Keras functional API**

      -  **Scratch via subclassing**

   - Keras 模型实现：

      - 类 Scikit-Learn API 示例：

         .. code:: python

            from tensorflow import keras
            from tensorflow.keras import layers, models
            from tensorflow.keras.datasets import mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0
            model = models.Sequential()
            model.add(layers.Dense(units = 64, activation = "relu"))
            model.add(layers.Dense(units = 10, activation = "softmax"))
            model.compile(
               loss = "categorical_crossentropy",
               optimizer = "sgd",
               metrics = ["accuracy"]
            )
            # model.compile(
            #     loss = keras.losses.categorical_crossentropy,
            #     optimizer = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
            # )
            model.fit(x_train, y_train, epochs = 5, batch_size = 32)
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)
            classes = model.predict(x_test, batch_size = 128)

      - 低级循环训练示例：

         .. code:: python

            import tensorflow as tf

            # prepare an optimizer.
            optimizer = tf.keras.optimizers.Adam()
            # prepare a loss function.
            loss_fn = tf.keras.losses.kl_divergence

            # Iterate over the batches of a dataset.
            for inputs, targets in dataset:
               # Open a GradientTape
               with tf.GradientTape() as tape:
                  # Forward pass.
                  predictions = model(inputs)
                  # Compute the loss value for this batch.
                  loss_value = loss_fn(targets, predictions)

               # Get gradients of loss wrt the weights.
               gradients = tape.gradient(loss_value, model.trainable_weights)
               # Update the weights of the model
               optimizer.apply_gradients(zip(gradients, model.trainable_weights))
