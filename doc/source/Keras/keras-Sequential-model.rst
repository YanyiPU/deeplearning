
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
