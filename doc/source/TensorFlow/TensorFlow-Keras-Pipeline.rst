
TensorFlow Keras Pipeline
==================================

1.Keras Sequential/Functional API 模式建立模型
-----------------------------------------------------------------

1.0 Keras Subclassing API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 使用 Keras 的 Subclassing API 建立模型，即对 ``tf.keras.Model`` 类进行扩展以定义自己的新模型

    .. code-block:: python
        
        import tensorflow as tf

        class MyModel(tf.keras.Model):
            def __init__(self):
                super.__init__()
                # 此处添加初始化的代码(包含call方法中会用到的层)例如：
                layer1 = tf.keras.layers.BuildInLayer()
                layer2 = MyCustomLayer(...)

            def call(self, input):
                # 此处添加模型调用的代码(处理输入并返回输出)，例如：
                x = layer1(input)
                output = layer2(x)
                return output

1.1 Sequential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python

        model = tf.keras.models.Sequential(
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation = tf.nn.relu),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Softmax()
        )

1.2 Functional API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python

        inputs = tf.keras.Input(shape = (28, 28, 1))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units = 10)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)

2.Keras Model--compile、fit、evaluate
-----------------------------------------------------------------

2.1 compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = tf.keras.losses.sparse_categorical_crossentropy,
            metrics = [tf.keras.metrics.sparse_categorical_accuracy]
        )

.. note:: 

    - ``tf.keras.Model.compile`` 3 个主要参数:

        - ``optimizer``: 优化器，可从 ``tf.keras.optimizers`` 中选择
            - .Adam(learning_rate)
        - ``loss``: 损失函数，可从 ``tf.keras.losses`` 中选择
            - .sparse_categorical_crossentropy
        - ``metrics``: 评估指标，可从 ``tf.keras.metrics`` 中选择
            - .sparse_categorical_accuracy

2.2 fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python
    
        model.fit(
            x = data_loader.train_data,
            y = data_loader.train_label,
            epochs = num_epochs,
            batch_size = batch_size,
            validation_data = data_loader.validation_data
        )

.. note:: 

    - ``tf.keras.Model.fit`` 5 个主要参数:

        - ``x``: 训练数据 
        - ``y``: 训练数据目标数据(数据标签)
        - ``epochs``: 将训练数据迭代多少遍
        - ``batch_size``: 批次的大小
        - ``validation_data``: 验证数据，可用于在训练过程中监控模型的性能

2.3 evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python

        print(model.evaluate(data_loader.test_data, data_loader.test_label))

.. note:: 

    - ``tf.keras.Model.evaluate`` 2 个参数:

        - ``x``: 测试数据
        - ``y``: 测试数据目标数据(数据标签)

3.自定义层、损失函数、评估指标
-----------------------------------------------------------------

3.1 自定义层
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 自定义层需要继承 ``tf.keras.layers.Layers`` 类，并重写 ``__init__``、``build``、``call`` 三个方法

    .. code-block:: python

        import numpy as np
        import tensorflow as tf

        class MyLayer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                # 初始化代码
            
            def build(self, input_shape): # input_shape 是一个 TensorShape 类型对象，提供输入的形状
                # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
                # 而不需要使用者额外指定变量形状
                # 如果已经可以完全确定变量的形状，也可以在 __init__ 部分创建变量
                self.variable_0 = self.add_weight(...)
                self.variable_1 = self.add_weight(...)
            
            def call(self, inputs):
                # 模型调用的代码(处理输入并返回输出)
                return output

- 线性层示例

    .. code-block:: python

        import numpy as np
        import tensorflow as tf

        class LinearLayer(tf.keras.layers.Layer):
            def __init__(self, units):
                super.__init__()
                self.units = units
            
            def build(self, input_shape):
                self.w = self.add_variable(
                    name = "w", 
                    shape = [input_shape[-1], self.units],  # [n, 1]
                    initializer = tf.zeros_initializer()
                )
                self.b = self.add_variable(
                    name = "b",
                    shape = [self.units],                   # [1]
                    initializer = tf.zeros_initializer()
                )
            
            def call(self, inputs):
                y_pred = tf.matmul(inputs, self.w) + self.b
                return y_pred
        
        class LinearModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.layer = LinearLayer(untis = 1)
            
            def call(self, inputs):
                output = self.layer(inputs)
                return output




3.2 自定义损失函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 自定义损失函数需要继承 ``tf.keras.losses.Loss`` 类，重写 ``call`` 方法即可，输入真实值 ``y_true`` 和模型预测值 ``y_pred``，
  输出模型预测值和真实值之间通过自定义的损失函数计算出的损失值

    .. code-block:: python  

        import numpy as np
        import tensorflow as tf

        class MeanSquaredError(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(tf.square(y_pred - y_true))

3.3 自定义评估指标
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 自定义评估指标需要继承 ``tf.keras.metrics.Metric`` 类，并重写 ``__init__``、``update_state``、``result`` 三个方法

    .. code-block:: python

        import numpy as np
        import tensorflow as tf

        class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
            def __init__(self):
                super().__init__()
                self.total = self.add_weight(name = "total", dtype = tf.int32, initializer = tf.zeros_initializer())
                self.count = self.add_weight(name = "total", dtype = tf.int32, initializer = tf.zeros_initializer())

            def update_state(self, y_true, y_pred, sample_weight = None):
                values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis = 1, output_type = tf.int32)), tf.int32)
                self.total.assign_add(tf.shape(y_true)[0])
                self.count.assign_add(tf.reduce_sum(values))

            def result(self):
                return self.count / self.total
