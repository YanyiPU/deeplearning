
Keras 损失函数
===============

    The purpose of loss functions is to compute the quantity that a model 
    should seek to minimize during training.

1.常用损失函数
------------------------------------------

    - class handle

        - 可以传递配置参数

    - function handle

1.1 概率损失(Probabilistic losses)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - ``BinaryCrossentropy`` class

        - ``binary_crossentropy()`` function

    - ``CategoricalCrossentropy`` class

        - ``categorical_crossentropy()`` function

    - ``SparseCategoricalCrossentropy`` class

        - ``sparse_categorical_crossentropy()`` function

    - ``Possion`` class

        - ``possion()`` function

    - ``KLDivergence`` class

        - ``kl_divergence()`` function

1.1.1 class & function() 使用方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 作用

    - 二分类损失函数

        - BinaryCrossentropy & binary_crossentropy

        - Computes the cross-entropy loss between true labels and predicted labels.

    - 二分类、多分类

        - CategoricalCrossentropy & categorical_crossentropy

        - SparseCategoricalCrossentropy & sparse_categorical_crossentropy
    
    - 其他

- 语法

    .. code-block:: python

        tf.keras.losses.Class(
            from_loits = False, 
            label_smoothing = 0, 
            reduction = "auto", 
            name = ""
        )

- 示例

    .. code-block:: python

        # data
        y_ture = [[0., 1.], [0., 0.]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]

        # reduction="auto" or "sum_over_batch_size"
        bce = tf.keras.losses.BinaryCrossentropy()
        bce(y_true, y_pred).numpy()
        
        # reduction=sample_weight
        bce = tf.keras.losses.BinaryCrossentropy()
        bce(y_true, y_pred, sample_weight = [1, 0]).numpy()

        # reduction=sum
        bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.SUM)
        bce(y_true, y_pred).numpy()

        # reduction=none
        bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
        bce(y_true, y_pred).numpy()

1.2 回归损失(Regression losses)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - ``MeanSquaredError`` class

        - ``mean_squared_error`` function 

    - ``MeanAbsoluteError`` class

        - ``mean_absolute_error`` function

    - ``MeanAbsolutePercentageError`` class

        - ``mean_absolute_percentage_error`` function

    - ``MeanSquaredLogarithmicError`` class

        - ``mean_squared_logarithmic_error`` function

    - ``CosineSimilarity`` class

        - ``cosine_similarity`` function

    - ``Huber`` class

        - ``huber`` function

    - ``LogCosh`` class

        - ``log_cosh`` function


1.3 Hinge losses for "maximum-margin" classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - ``Hinge`` class

        - ``hinge`` function

    - ``SquaredHinge`` class

        - ``squared_hinge`` function

    - ``CategoricalHinge`` class

        - ``categorical_hinge`` function


2.损失函数的使用——compile() & fit()
------------------------------------------

    - 通过实例化一个损失类创建损失函数，可以传递配置参数

        .. code-block:: python

            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential()
            model.add(layers.Dense(64, kernel_initializer = "uniform", input_shape = (10,)))
            model.add(layers.Activation("softmax"))
            
            model.compile(
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                optimizer = "adam", 
                metrics = ["acc"]
            )

    - 直接使用损失函数

        .. code-block:: python

            from tensorflow.keras.losses import sparse_categorical_crossentropy

            model.compile(
                loss = "sparse_categorical_crossentropy", 
                optimizer = "adam", 
                metrics = ["acc"]
            )


3.损失函数的使用——单独使用
------------------------------------------

.. code-block:: python

    tf.keras.losses.mean_squared_error(tf.ones((2, 2)), tf.zeros((2, 2)))
    loss_fn = tf.keras.losses.MeanSquaredError(resuction = "sum_over_batch_size")
    loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

    loss_fn = tf.keras.losses.MeanSquaredError(reduction = "sum")
    loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

    loss_fn = tf.keras.losses.MeanSquaredError(reduction = "none")
    loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

    loss_fn = tf.keras.losses.mean_squared_error
    loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))


4.创建自定义损失函数
------------------------------------------

    -  Any callable with the signature ``loss_fn(y_true, y_pred)`` that returns an array of 
       losses (one of sample in the input batch) can be passed to compile() as a loss. 
    
    - Note that sample weighting is automatically supported for any such loss.

示例：

    .. code-block:: python
    
        def my_loss_fn(y_true, y_pred):
            squared_difference = tf.square(y_true - y_pred)
            return tf.reduce_mean(squared_difference, axis = -1)

        model.compile(optimizer = "adam", loss = my_loss_fn)


5. ``add_loss()`` API
------------------------------------------

.. code-block:: python

    from tensorflow.keras.layers import Layer

    class MyActivityRegularizer(Layer):
        """Layer that creates an activity sparsity regularization loss."""

        def __init__(self, rate = 1e-2):
            super(MyActivityRegularizer, self).__init__()
            self.rate = rate

        def call(self, inputs):
            self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))

            return inputs
    
    from tensorflow.keras import layers

    class SparseMLP(Layer):
        """Stack of Linear layers with a sparsity regularization loss."""

        def __init__(self, output_dim):
            super(SparseMLP, self).__init__()
            self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
            self.regularization = MyActivityRegularizer(1e-2)
            self.dense_2 = layers.Dense(output_dim)

        def call(self, inputs):
            x = self.dense_1(inputs)
            x = self.regularization(x)
            return self.dense_2(x)

    mlp = SparseMLP(1)
    y = mlp(tf.ones((10, 10)))

    print(mlp.losses)  # List containing one float32 scalar

    mlp = SparseMLP(1)
    mlp(tf.ones((10, 10)))
    assert len(mlp.losses) == 1
    mlp(tf.ones((10, 10)))
    assert len(mlp.losses) == 1  # No accumulation.





