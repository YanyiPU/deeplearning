

keras Functional API
====================

API 使用总结:

   - keras.models.load_model
   
   - .save()


Functional API 的使用技巧：

   - 优势

   - 弱点



入门
------

库文件导入：

   .. code:: python

      import numpy as np 
      import tensorflow as tf 
      from tensorflow import keras
      from tensorflow.keras import layers

模型拓扑结构:

   .. code-block:: 

      (input: 784-dimensional vectors)
            ↧
      [Dense (64 units, relu activation)]
            ↧
      [Dense (64 units, relu activation)]
            ↧
      [Dense (10 units, softmax activation)]
            ↧
      (output: logits of a probability distribution over 10 classes)

Functional API 模型:

   .. code-block:: python

      >>> from tensorflow import keras
      >>> from tensorflow.keras import layers, models
      >>> inputs = keras.Input(shape = (784,))
      >>> x = layers.Dense(64, activation = "relu")(inputs)
      >>> x = layers.Dense(64, activation = "relu")(x)
      >>> outputs = layers.Dense(10)(x)
      >>> model = keras.Model(inputs = inputs, outputs = outputs, name = "mnist_model")
      >>> model.summary()
      Model: "mnist_model"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      input_1 (InputLayer)         [(None, 784)]             0         
      _________________________________________________________________
      dense (Dense)                (None, 64)                50240     
      _________________________________________________________________
      dense_1 (Dense)              (None, 64)                4160      
      _________________________________________________________________
      dense_2 (Dense)              (None, 10)                650       
      =================================================================
      Total params: 55,050
      Trainable params: 55,050
      Non-trainable params: 0
      _________________________________________________________________
      >>> print(inputs.shape)
      >>> print(inputs.dtype)
      (None, 784)
      <dtype: 'float32'>
      >>> keras.utils.plot_model(model, "my_first_model.png")
      >>> keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes = True)


1.Training, Evaluation, Inference
--------------------------------------

使用 API:

   - .compile()

      - keras.losses.SparseCategoricalCrossentropy()

      - keras.optimizer.RMSprop()

   - .fit()

   - .evaluate()


步骤：

   - 编译

   - 训练

   - 评估


.. code-block:: python

   # mnist data
   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   x_train = x_train.reshape(60000, 784).astype("float32") / 255
   x_test = x_test.reshape(10000, 784).astype("float32") / 255

   # model compile
   model.compile(
      loss = keras.SparseCategoricalCrossentropy(from_logits = True),
      optimizer = keras.optimizer.RMSprop(),
      metrics = ["accuracy"],
   )

   # model train
   history = model.fit(x_train, y_train, batch_size = 64, epochs = 2, validation_split = 0.2)

   # model evaluate
   test_scores = model.evaluate(x_test, y_test, verbose = 2)
   print("Test loss:", test_scores[0])
   print("Test accuracy:", test_scores[1])

2.Save, Serialize
--------------------------------------

使用 API:

   - .save()

   - keras.model.load_model()


保存内容：

   - model architecture

   - model weight values

   - model train config

   - optimizer and its state(as passed to compile)

   - to restart training where left off


.. code-block:: python

   model.save("path_to_my_model")
   del model
   # Recreate the exact same model purely from the file 
   model = keras.models.load_model("path_to_my_model")


3.模型网络层共享
--------------------------------------

3.1 网络层共享
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Encoder
   encoder_input = keras.Input(shape = (28, 28, 1), name = "img")
   x = layers.Conv2D(16, 3, activation = "relu")(encoder_input)
   x = layers.Conv2D(32, 3, activation = "relu")(x)
   x = layers.MaxPooling2D(3)(x)
   x = layers.Conv2D(32, 3, activation = "relu")(x)
   x = layers.Conv2D(16, 3, activation = "relu")(x)
   encoder_output = layers.GlobalMaxPooling2D()(x)
   encoder = keras.Model(encoder_input, encoder_output, name = "encoder")
   encoder.summary()

   # decoder
   x = layers.Reshape((4, 4, 1))(encoder_output)
   x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
   x = layers.Conv2DTranspose(32, 3, activation = "relu")(x)
   x = layers.UpSampling2D(3)(x)
   x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
   decoder_output = layers.Conv2DTranspose(1, 3, activation = "relu")(x)

   # AutoEncoder
   autoencoder = keras.Model(encoder_input, decoder_output, name = "autoencoder")
   autoencoder.summary()


3.2 模型共享
~~~~~~~~~~~~~~~~~~~~

a model is just like a layer.

示例 1: AutoEncoder

.. code-block:: python

   # Encoder
   encoder_input = keras.Input(shape = (28, 28, 1), name = "original_img")
   x = layers.Conv2D(16, 3, activation = "relu")(encoder_input)
   x = layers.Conv2D(32, 3, activation = "relu")(x)
   x = layers.MaxPooling2D(3)(x)
   x = layers.Conv2D(32, 3, activation = "relu")(x)
   x = layers.Conv2D(16, 3, activation = "relu")(x)
   encoder_output = layers.GlobalMaxPooling2D()(x)
   encoder = keras.Model(encoder_input, encoder_output, name = "encoder")
   encoder.summary()

   # decoder
   decoder_input = keras.Input(shape = (16,), name = "encoded_img")
   x = layers.Reshape((4, 4, 1))(decoder_input)
   x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
   x = layers.Conv2DTranspose(32, 3, activation = "relu")(x)
   x = layers.UpSampling2D(3)(x)
   x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
   decoder_output = layers.Conv2DTranspose(1, 3, activation = "relu")(x)
   decoder = keras.Model(decoder_input, decoder_output, name = "decoder")
   decoder.summary()

   # AutoEncoder
   autoencoder_input = keras.Input(shape = (28, 28, 1), name = "img")
   encoded_img = encoder(autoencoder_input)
   decoded_img = decoder(encoded_img)
   autoencoder = keras.Model(autoencoder_input, decoded_img, name = "autoencoder")
   autoencoder.summary()


示例 2: Ensembling

.. code-block:: python

   def get_model():
      inputs = keras.Input(shape = (128,))
      outputs = keras.Dense(1)(inputs)

      return keras.Model(inputs, outputs)

   model1 = get_model()
   model2 = get_model()
   model3 = get_model()

   inputs = keras.Input(shape = (128,))
   y1 = model1(inputs)
   y2 = model2(inputs)
   y3 = model3(inputs)
   outputs = layers.average([y1, y2, y3])

   ensemble_model = keras.Model(inputs = inputs, outputs = outputs)


3.3 复杂拓扑图模型
~~~~~~~~~~~~~~~~~~~~

   - 模型有多个输入、输出

   - ResNet

示例 1:

.. code-block:: python





4.自定义层扩展 API
---------------------------



5.Sequential API、Functional API、Model subclassing API 混搭
--------------------------------------------------------------

.. code-block:: python

   units = 32
   timesteps = 10
   input_dim = 4

   # Define a Functional model
   inputs = keras.Input(shape = (None, units))
   x = layers.GloabalAveragePooling1D()(inputs)
   outputs = layers.Dense(1)()
   model = keras.Model(inputs, outputs)

   # Define a subclassing model
   class CustomRNN(layres.Layer):

      def __init__(self):
         super(CustomRNN, self).__init__()
         self.units = units
         self.projection_1 = layers.Dense(units = units, activation = "tanh")
         self.projection_2 = layers.Dense(units = units, activation = "tanh")
         # Our previously-defined Functional model
         self.classifier = model
      
      def call(self, inputs):
         outputs = []
         state = tf.zeros(shape = (inputs.shape[0], self.units))
         for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
         features = tf.stack(outputs, axis = 1)
         print(features.shape)
         
         return self.classifier(features)
   run_model = CustomRNN()
   _ = run_model(tf.zeros((1, timesteps, input_dim)))

