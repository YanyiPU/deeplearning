
Keras 后端
==========

1.什么是 Keras 后端？
-----------------------

   Keras 后端:

      Keras 是一个模型级库, 为开发深度学习模型提供了高层次的构建模块。它不处理诸如张量乘积和卷积等低级操作。
      相反, 它依赖于一个专门的、优化的张量操作库来完成这个操作, 它可以作为 Keras 的「后端引擎」。
      相比单独地选择一个张量库, 而将 Keras 的实现与该库相关联, Keras 以模块方式处理这个问题, 
      并且可以将几个不同的后端引擎无缝嵌入到 Keras 中。

   目前可用的 Keras 后端:

      -  TensorFlow

      -  Theano

      -  CNTK


2.从一个后端切换到另一个后端
----------------------------

   如果您至少运行过一次 Keras, 您将在以下位置找到 Keras 配置文件. 如果没有, 可以手动创建它.

   Keras 配置文件位置:

      .. code:: shell

         # Liunx or Mac
         $ vim $HOME/.keras/keras.json

         # Windows
         $ vim %USERPROFILE%/.keras/keras.json

   Keras 配置文件创建:

      .. code:: shell

         $ cd ~/.keras
         $ sudo subl keras.json

   也可以定义环境变量 ``KERAS_BACKEND``, 不过这会覆盖配置文件 ``$HOME/.keras/keras.json`` 中定义的内容:

      .. code:: 

         KERAS_BACKEND=tensorflow python -c "from keras import backend" 
         Using TensorFlow backend.

   当前环境的 Keras 配置文件内容:

      .. code:: json

         {
            "floatx": "float32",
            "epsilon": 1e-07,
            "backend": "tensorflow",
            "image_data_format": "channels_last"
         }

   自定义 Keras 配置文件:

      -  在 Keras 中, 可以加载除了 "tensorflow", "theano" 和 "cntk"
         之外更多的后端。Keras 也可以使用外部后端, 这可以通过更改 keras.json
         配置文件和 "backend" 设置来执行。 假设您有一个名为 my_module 的 Python
         模块, 您希望将其用作外部后端。keras.json 配置文件将更改如下.

         - 必须验证外部后端才能使用, 有效的后端必须具有以下函数:

            -  ``placeholder``
            -  ``variable``
            -  ``function``

         - 如果由于缺少必需的条目而导致外部后端无效, 则会记录错误, 通知缺少哪些条目:

            .. code:: shell

               {
                  "image_data_format": "channels_last",
                  "epsilon": 1e-07,
                  "floatx": "float32",
                  "backend": "my_package.my_module"
               }

3.keras.json 详细配置
---------------------

   -  ``image_data_format``:

      -  ``"channels_last"``

         -  (rows, cols, channels)

         -  (conv*dim1, conv*\ dim2, conv_dim3, channels)

      -  ``"channels_first"``

         -  (channels, rows, cols)

         -  (channels, conv\ *dim1, conv*\ dim2, conv_dim3)

      -  在程序中返回: ``keras.backend.image_data_format()``

   -  ``epsilon``:

      -  浮点数, 用于避免在某些操作中被零除的数字模糊常量

   -  ``floatx``:

      -  字符串: ``float16``, ``float32``, ``float64``\ 。默认浮点精度

   -  ``backend``:

      -  字符串: ``tensorflow``, ``theano``, ``cntk``

4.使用抽象 Keras 后端编写新代码
-------------------------------

如果你希望你编写的 Keras 模块与 Theano (th) 和 TensorFlow (tf) 兼容,
则必须通过抽象 Keras 后端 API 来编写它们。

   .. code:: python

      from keras import backend as K
      import numpy as np

      # 实例化一个输入占位符
      inputs = K.placeholder(shape = (2, 4, 5))
      inputs = K.placeholder(shape = (None, 4, 5))
      inputs = K.placeholder(ndim = 3)

      # 实例化一个变量
      val = np.random.random((3, 4, 5))
      var = K.variable(value = val)
      var = K.zeros(shape = (3, 4, 5))
      var = K.ones(shape = (3, 4, 5))

等价于：

   .. code:: python

      import tensorflow as tf

      # 实例化一个输入占位符
      inputs = tf.placeholder()
      inputs = tf.tensor.matrix()
      inputs = tf.tensor.tensor3()

      # 实例化一个变量
      var = tf.Variable()
      var = tf.shared()

5.后端函数
----------

   -  ``keras.backend.backend()``

      -  Keras 目前正在使用的后端名

   -  ``keras.backend.symbolic(func)``

      -  Decorator used in TensorFlow 2.0 to enter the Keras graph.

   -  ``keras.backend.eager(func)``

      -  Decorator used in TensorFlow 2.0 to exit the Keras graph.

   -  ``keras.backend.get_uids(prefix = "")``

      -  获取默认计算图的标识符
      -  prefix: 图的可选前缀

   -  ``keras.backend.manual_variable_initialization(value)``

      -  设置变量手动初始化标志

   -  ``keras.backend.epsilon()``

      -  Returns the value of the fuzz factor used in numeric expressions.

   -  ``keras.backend.reset_uids()``

      -  重置图的标识符

6.Resets graph identifiers
--------------------------

   -  ``keras.backend.set_epsilon(e)``

   -  ``keras.backend.floatx()``

      -  ``keras.backend.set_floatx()``
      -  ``keras.backend.cast_to_floatx()``

   -  ``keras.backend.image_data_format()``

      -  ``keras.backend.set_image_data_format(data_format)``

   -  ``keras.backend.learning_phase()``

      -  ``keras.backend.set_learning_phase()``

   -  ``keras.backend.clear_session()``

      -  销毁当前的 TF 图并创建一个新图
      -  有用于避免旧模型/网络层混乱

   -  张量(Tensor)

      -  ``keras.backend.is_sparse()``
      -  ``keras.backend.to_dense()``
      -  ``keras.backend.variable()``
      -  ``keras.backend.constant()``
      -  ``keras.backend.is_keras_tensor()``
      -  ``keras.backend.is_tensor()``
      -  ``keras.backend.placeholder()``
      -  ``keras.backend.is_placeholder()``
      -  ``keras.backend.shape()``
      -  ``keras.backend.int_shape()``
      -  ``keras.backend.ndim()``
      -  ``keras.backend.dtype()``
      -  ``keras.backend.eval()``
      -  ``keras.backend.zeros()``
      -  ``keras.backend.zeros_like()``
      -  ``keras.backend.ones()``
      -  ``keras.backend.ones_like()``
      -  ``keras.backend.eye()``
      -  ``keras.backend.identity()``
      -  ``keras.backend.random_uniform_variable()``
      -  ``keras.backend.random_normal_variable()``
      -  ``keras.backend.count_params()``
      -  ``keras.backend.cast()``
      -  ``keras.backend.update()``
      -  ``keras.backend.update_add()``
      -  ``keras.backend.update_sub()``
      -  ``keras.backend.moving_average_update()``
      -  ``keras.backend.batch_dot()``
      -  ``keras.backend.transpose()``
      -  ``keras.backend.gather()``
      -  ``keras.backend.max()``
      -  ``keras.backend.min()``
      -  ``keras.backend.sum()``
      -  ``keras.backend.prod()``
      -  ``keras.backend.cumsum()``
      -  ``keras.backend.cumprod()``
      -  ``keras.backend.var()``
      -  ``keras.backend.std()``
      -  ``keras.backend.mean()``
      -  ``keras.backend.any()``
      -  ``keras.backend.all()``
      -  ``keras.backend.argmax()``
      -  ``keras.backend.argmin()``
      -  ``keras.backend.square()``
      -  ``keras.backend.abs()``
      -  ``keras.backend.sqrt()``
      -  ``keras.backend.exp()``
      -  ``keras.backend.log()``
      -  ``keras.backend.logsumexp()``
      -  ``keras.backend.round()``
      -  ``keras.backend.sign()``
      -  ``keras.backend.pow()``
      -  ``keras.backend.clip()``
      -  ``keras.backend.equal()``
      -  ``keras.backend.not_equal()``
      -  ``keras.backend.greater()``
      -  ``keras.backend.greater_equal()``
      -  ``keras.backend.less()``
      -  ``keras.backend.less_equal()``
      -  ``keras.backend.maximum()``
      -  ``keras.backend.minimum()``
      -  ``keras.backend.sin()``
      -  ``keras.backend.cos()``
      -  ``keras.backend.normalize_batch_in_training()``
      -  ``keras.backend.batch_normalization()``
      -  ``keras.backend.concatenate()``
      -  ``keras.backend.reshape()``
      -  ``keras.backend.permute_dimensions()``
      -  ``keras.backend.resize_images()``
      -  ``keras.backend.resize_volumes()``
      -  ``keras.backend.repeat_elements()``
      -  ``keras.backend.repeat()``
      -  ``keras.backend.arange()``
      -  ``keras.backend.tile()``
      -  ``keras.backend.flatten()``
      -  ``keras.backend.batch_flatten()``
      -  ``keras.backend.expand_dims()``
      -  ``keras.backend.squeeze()``
      -  ``keras.backend.temporal_padding()``
      -  ``keras.backend.spatial_2d_padding()``
      -  ``keras.backend.spatial_3d_padding()``
      -  ``keras.backend.stack()``
      -  ``keras.backend.one_hot()``
      -  ``keras.backend.reverse()``
      -  ``keras.backend.slice()``
      -  ``keras.backend.get_value()``
      -  ``keras.backend.batch_get_value()``
      -  ``keras.backend.set_value()``
      -  ``keras.backend.batch_set_value()``
      -  ``keras.backend.print_tensor()``
      -  ``keras.backend.function()``
      -  ``keras.backend.gradients()``
      -  ``keras.backend.stop_gradient()``
      -  ``keras.backend.rnn()``
      -  ``keras.backend.switch()``
      -  ``keras.backend.in_train_phase()``
      -  ``keras.backend.in_test_phase()``
      -  ``keras.backend.relu()``
      -  ``keras.backend.elu()``
      -  ``keras.backend.softmax()``
      -  ``keras.backend.softplus()``
      -  ``keras.backend.softsign()``
      -  ``keras.backend.categorical_crossentropy()``
      -  ``keras.backend.sparse_categorical_crossentropy()``
      -  ``keras.backend.binary_crossentropy()``
      -  ``keras.backend.sigmoid()``
      -  ``keras.backend.hard_sigmoid()``
      -  ``keras.backend.tanh()``
      -  ``keras.backend.dropout()``
      -  ``keras.backend.l2_normalize()``
      -  ``keras.backend.in_top_k()``
      -  ``keras.backend.conv1d()``
      -  ``keras.backend.conv2d()``
      -  ``keras.backend.conv2d_transpose()``
      -  ``keras.backend.separable_conv1d()``
      -  ``keras.backend.separable_conv2d()``
      -  ``keras.backend.depthwise_conv2d()``
      -  ``keras.backend.conv3d()``
      -  ``keras.backend.conv3d_transpose()``
      -  ``keras.backend.pool2d()``
      -  ``keras.backend.pool3d()``
      -  ``keras.backend.bias_add()``
      -  ``keras.backend.random_normal()``
      -  ``keras.backend.random_uniform()``
      -  ``keras.backend.random_binomial()``
      -  ``keras.backend.truncated_normal()``
      -  ``keras.backend.ctc_label_dense_to_sparse()``
      -  ``keras.backend.ctc_batch_cost()``
      -  ``keras.backend.ctc_decode()``
      -  ``keras.backend.map_fn()``
      -  ``keras.backend.foldl()``
      -  ``keras.backend.foldr()``
      -  ``keras.backend.local_conv1d()``
      -  ``keras.backend.local_conv2d()``
