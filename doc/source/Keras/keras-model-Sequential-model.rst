
keras Sequential model
=============================

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

1.什么时候使用 Sequential 模型
------------------------------------------

A Sequential model is appropriate for a plain stack of 
layers where each layer has exactly one input tensor 
and one output tensor.

A Sequential model is not appropriate when:

    - Your model has multiple inputs or multiple outputs
    - Any of your layers has multiple inputs or multiple outputs
    - You need to do layer sharing
    - You want non-linear topology (e.g. a residual connection, a multi-branch model)

2.构建一个 Sequential 模型
------------------------------------------

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential(name = "my_sequential")
    model.add(layers.Dense(2, activation = "relu", name = "layer1"))
    model.add(layers.Dense(3, activation = "relu", name = "layer2"))
    model.add(layers.Dense(4, name = "layer3"))
    print(model.layers)

    # delete latest layer of the model
    model.pop()
    print(len(model.layers))




4.Sequential 模型的特征提取功能
-------------------------------------------



5.Sequential 模型实现 Transfer images
--------------------------------------------

Transfer learning consists of freezing the bottom layers in a model and only 
training the top layers. If you aren't familiar with it, make sure to read our 
guide to transfer learning.

Here are two common transfer learning blueprint involving Sequential models.

- First, let's say that you have a Sequential model, and you want to freeze all 
  layers except the last one. In this case, you would simply iterate over ``model.
  layers`` and set ``layer.trainable = False`` on each layer, except the last one. Like this:

.. code-block:: python

    # Sequential model
    model = keras.Sequential()
    model.add(keras.Input(shape = (784))
    model.add(layers.Dense(32, activation = "relu"))
    model.add(layers.Dense(32, activation = "relu"))
    model.add(layers.Dense(32, activation = "relu"))
    model.add(layers.Dense(10))

    # Presumably you would want to first load pre-trained weights
    model.load_weights(...)

    # Freeze all layers except the last one
    for layer in model.layers[:-1]:
        layer.trainable = False
    
    # Recompile and train(this will only update the weights of the last layer).
    model.compile(...)
    model.fit(...)

- Another common blueprint is to use a Sequential model to stack a pre-trained model and some 
  freshly initialized classification layers. Like this:

.. code-block:: python

    # Load a convolutional base with pre-trained weights
    base_model = keras.applications.Xception(
        weights = "imagenet",
        include_top = False,
        pooling = "avg"
    )

    # Freeze the base model
    base_model.trainable = False