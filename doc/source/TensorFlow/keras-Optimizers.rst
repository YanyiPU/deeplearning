
Keras 优化器
===============


1.常用 Keras 优化器
---------------------------------------------

    - SGD
    - RMSprop
    - Adam
    - Adadelta
    - Adagrad
    - Adamax
    - Nadam
    - Ftrl


1.1 TODO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



2.Keras 优化器的使用
--------------------------------------------

2.1 模型编译(compile)和拟合(fit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python

        from tensorflow import keras
        from tensorflow.keras import layers
        
        # model
        model = keras.Sequential()
        model.add(layers.Dense(64, kernel_initializer = "uniform", input_shape = (10,)))
        model.add(layers.Activate("softmax"))
        # model compile
        opt = keras.optimizers.Adam(learning_rate = 0.01)
        model.compile(loss = "categorical_crossentropy", optimizer = opt)
        # model.compile(loss = "categorical_crossentropy", optimizer = "adam")

2.2 自定义迭代训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python

        # Instantiate an optimizer
        optimizer = tf.keras.optimizer.Adam()

        # Iterate over the batches of a dataset.
        for x, y in dataset:
            # open a GradientTape
            with tf.GradientTape() as tape:
                
                # Forward pass.
                logits = model(x)
                
                # Loss value for this batch
                loss_value = loss_fn(y, logits)
            
            # Get gradients of loss wrt the weights
            gradients = tape.gradient(loss_value, model.trainable_weights)

            # Update the weights of the model
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))


2.3 学习率衰减(decay)、调度(sheduling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 可以使用学习率时间表来调整优化器的学习率如何随时间变化

        - ExponentialDecay: 指数衰减
        - PiecewiseConstantDecay: 
        - PolynomialDecay： 多项式衰减
        - InverseTimeDecay: 逆时间衰减

    .. code-block:: python

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 1e-2,
            decay_steps = 10000,
            decay_rate = 0.9
        )
        optimizer = keras.optimizers.SGD(learning_rate = lr_schedule)


3.Keras 优化算法核心 API
--------------------------------------------

    - apply_gradients
    - weights_property
    - get_weights
    - set_weights

3.1 apply_gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 语法

        .. code-block:: python

            Optimizer.apply_gradients(
                grads_and_vars, name=None, experimental_aggregate_gradients=True
            )

    - 参数

        - grads_and_vars: 梯度、变量对的列表
        - name: 返回的操作的名称
        - experimental_aggregate_gradients: 

    - 示例

        .. code-block:: python

            grads = tape.gradient(loss, vars)
            grads = tf.distribute.get_replica_context().all_reduce("sum", grads)

            # Processing aggregated gradients.
            optimizer.apply_gradients(zip(grad, vars), experimental_aggregate_gradients = False)


3.2 weights_property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 语法

        .. code-block:: python

            import tensorflow as tf

            tf.keras.optimizers.Optimizer.weights


3.3 get_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 语法

        .. code-block:: python

            Optimizer.get_weights()

    - 示例

        .. code-block:: python

            # 模型优化器
            opt = tf.keras.optimizers.RMSprop()

            # 模型构建、编译
            m = tf.keras.models.Sequential()
            m.add(tf.keras.layers.Dense(10))
            m.compile(opt, loss = "mse")
            
            # 数据
            data = np.arange(100).reshape(5, 20)
            labels = np.zeros(5)

            # 模型训练
            print("Training")
            results = m.fit(data, labels)
            print(opt.get_weights)

3.4 set_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 语法

        .. code-block:: python
        
            Optimizer.set_weights(weights)


    - 示例

        .. code-block:: python

            # 模型优化器
            opt = tf.keras.optimizers.RMSprop()

            # 模型构建、编译
            m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
            m.compile(opt, loss = "mse")

            # 数据        
            data = np.arange(100).reshape(5, 20)
            labels = np.zeros(5)
            
            # 模型训练
            print("Training")
            results = m.fit(data, labels)

            # 优化器新权重
            new_weights = [
                np.array(10),       # 优化器的迭代次数
                np.ones([20, 10]),  # 优化器的状态变量
                np.zeros([10])      # 优化器的状态变量
            ]
            opt.set_weights(new_weights)
            opt.iteration
