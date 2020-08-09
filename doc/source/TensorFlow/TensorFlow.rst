
TensorFlow
==============

TensorFlow for beginners
--------------------------------------

1.1 Tensorflow 使用 [version old]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  创建/定义:

   -  张量(tf.Tensor)

   -  变量(tf.Variable)

   -  占位符变量(tf.placeholder)

   -  张量之间的运算操作

-  初始化张量

-  创建会话

-  执行会话

.. code:: python

   import tensorflow as tf

   # 定义两个常量 y_hat, y
   y_hat = tf.constant(36, name = 'y_hat')        
   y = tf.constant(39, name = 'y')
   # 定义一个变量 loss
   loss = tf.Variable((y - y_hat) ** 2, name = 'loss')

   # 对变量初始化
   init = tf.global_variables_initializer() 
          
   # 创建计算会话，执行会话
   with tf.Session() as session:
       session.run(init)                         
       print(session.run(loss))


1.2 神经网络基础函数
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   import tensorflow as tf

   def linear_function():
       """
       # 线性函数: y = w * x + b.
       """
       np.random.seed(1)
       # 定义张量
       X = tf.constant(np.random.randn(3, 1), name = "X")
       W = tf.constant(np.random.randn(4, 3), name = "W")
       b = tf.constant(np.random.randn(4, 1), name = "b")
       Y = tf.add(tf.matmul(W, X), b)
       # 初始化张量
       init = tf.global_variables_initializer()
       # 创建计算会话，执行会话
       with tf.Session() as session:
           session.run(init)
           result = session.run(Y)
       return result


   def sigmoid(z):
       """
       # Sigmoid function.
       """
       x = tf.placeholder(tf.float32, name = "x")
       sigmoid = tf.sigmoid(x)
       with tf.Session() as session:
           result = session.sun(sigmoid, feed_dict = {x: z})
       return result


   def crossentropy_loss(logits, labels):
       """
       # compute the cost using the sigmoid cross entropy.
       """
       z = tf.palceholder(tf.float32, name = "z")
       y = tf.placeholder(tf.float32, name = "y")
       cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, lables = y)
       with tf.Session() as session:
           cost = session.run(cost, 
                              feed_dict = {
                                  z: logits, 
                                  y: labels
                              })
       return cost


   def ont_hot_matrix(labels, C):
       """
       # One-Hot Encoding
       Arguments:
           labels: vector containing the labels
           C     : number of classes, the depth of the one hot dimension
       Returns:
           one_hot: one-hot matrix
       """
       C = tf.constant(C)
       one_hot_matrix = tf.one_hot(labels, C, axis = 0)
       with tf.Session() as session:
           one_hot = session.run(one_hot_matrix)
       return one_hot


   def ones(shape):
       """
       # create an array of ones of dimension shape
       # Arguments:
           shape: shape of the array
       Returns: 
           ones: array containing only ones
       """
       ones = tf.ones(shape)
       with tf.Session() as session:
           ones = session.run(ones)
       return ones

1.3 简单的神经网路模型搭建
~~~~~~~~~~~~~~~~~~~~~~~~~~

简单的手写数字(mnist)分类

.. code:: python

   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Flatten, Dense, Dropout

   # 导入数据
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()

   # 数据与处理(标准化)
   x_train = x_train / 255.0
   x_test = x_test / 255.0

   # 建立模型
   model = Sequential([
       Flatten(input_shape = (28, 28)),
       Dense(128, activation = 'relu'),
       Dropout(0.2),
       Dense(10, activation = 'softmax')
   ])

   # 编译模型
   model.compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'])

   # 训练模型
   model.fit(x_train, y_train, epochs = 5)

   # 评估模型
   model.evaluate(x_test, y_test)

   # 模型预测
   model.predict(test_images)

简单的衣服数据(fashion_mnist)分类

.. code:: python

   # 导入库
   import tensorflow as tf
   from tensorflow.model import Sequential
   from tensorflow.layers import Flatten, Dense
   from tensorflow.nn import relu, softmax
   from tensorflow.train import AdamOptimizer
   import numpy as np

   # 导入数据
   fashion_mnist = tf.keras.datasets.fashion_mnist
   (trian_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

   # 数据预处理
   class_names = [
       'T-shirt/top',  # 0
       'Trouser',      # 1
       'Pullover',     # 2
       'Dress',        # 3
       'Coat',         # 4
       'Sandal',       # 5
       'Shirt',        # 6
       'Sneaker',      # 7
       'Bag',          # 8
       'Ankle boot'    # 9
   ]
   train_images = trian_images / 255.0
   test_images = test_images / 255.0

   # 设置网络层
   model = Sequential([
       Flatten(input_shape = (28, 28)),
       Dense(128, activation = relu),
       Dense(10, activation = softmax)
   ])

   # 编译模型
   model.compile(
       optimizer = AdamOptimizer(),
       loss = "sparse_categorical_crossentropy",
       metrics = ['accuracy']
   )
    
   # 训练模型
   model.fit(train_images, train_labels, epochs = 5)

   # 模型评估
   test_loss, test_acc = model.evaluate(test_images, test_labels)
   print("Test accuracy:", test_acc)

   # 模型预测
   predictions = model.predict(test_images)
   print(predictions)
   max_prob_index = np.argmax(predictions[0])
   print(max_prob_index)
   print(test_labels[0])



2.TensorFlow 低阶 API (TensorFlow Core)
----------------------------------------

   -  管理 TensorFlow 程序: ``tf.Graph`` 、 TensorFlow 会话:
      ``tf.Session``\ 。而不是依靠 Estimator 来管理

   -  使用 ``tf.Session`` 运行 TensorFlow 操作

   -  在低级别环境中使用高级别组件:

      -  datasets

      -  layers

      -  feature_columns

   -  构建自己的训练循环，而不是使用 Estimator 提供的训练循环

.. code:: python

   from __future__ import absolute_import
   from __future__ import division
   from __future__ import print_function

   import numpy as np
   import tensorflow as tf

2.1 张量
~~~~~~~~

   -  TensorFlow 中的核心数据单元是张量(tensor)，表示为基本数据类型的 n
      维数组(array)

      -  tensor 的\ **阶(rank)**\ 是张量的维数(dim)

      -  tensor 的\ **形状(shape)**\ 是一个指定了 tensor 中每个 dim 中
         array 的 length 的整数元组(tuple)

      -  tensor 中的每个元素都具有形同的数据类型

      -  TensorFlow 使用 Numpy arrays 表示 tensor

   -  TensorFlow 操作和传递的主要对象是 ``tf.Tensor``

      -  TensorFlow 程序首先会构建一个 ``tf.Tensor``
         对象图(Graph)，详细说明如何基于其他可用 tensor 计算每个
         tensor，然后运行 计算图(Graph) 获得结果

   -  ``tf.Tensor`` 属性

      -  数据类型(data type)

      -  形状(shape)

   -  特殊张量：

      -  ``tf.Variable``

      -  ``tf.constant``

      -  ``tf.placeholder``

      -  ``tf.SparseTensor``

   -  ``tf.Tensor`` 的数据类型

      -  ``tf.string``

      -  ``tf.int16``

      -  ``tf.int32``

      -  ``tf.int64``

      -  ``tf.float16``

      -  ``tf.float32``

      -  ``tf.float64``

      -  ``tf.complex64``

      -  ``tf.bool``

**APIs：**

-  ``tf.Variable()``

-  ``tf.constant()``

-  ``tf.placeholder()``

-  ``tf.SparseTensor()``

-  ``tf.ones()``

-  ``tf.zeros()``

-  ``tf.rank()``

-  ``.shape``, ``tf.shape()``

-  ``tf.reshape(Tensor, [])``

-  ``.dtype``

-  ``tf.cast(Tensor, dtype)``

-  ``.eval()``

-  ``tf.Print()``


2.1.1 阶(rank)
^^^^^^^^^^^^^^^^^^^^^

``tf.Tensor``\ 对象的阶(rank)是它本身的维数。TensorFlow中的阶与数学矩阵的阶并不是同一个概念：

+----+-------------------+
| 阶 | 数学实例          |
+====+===================+
| 0  | 标量(只有大小)    |
+----+-------------------+
| 1  | 矢量(大小和方向)  |
+----+-------------------+
| 2  | 矩阵(数据表)      |
+----+-------------------+
| 3  | 3阶张量(数据块)   |
+----+-------------------+
| n  | n阶张量(自行想象) |
+----+-------------------+

**0 阶张量：**

.. code:: python

   mammal = tf.Variable("Elephant", tf.string)

   ignition = tf.Variable(451, tf.int16)

   floating = tf.Variable(3.14159265359, tf.float64)

   its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

**1 阶张量：**

-  可以传递一个项目列表作为初始值

.. code:: python

   mystr = tf.Variable(["Hello"], tf.string)

   cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)

   first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)

   its_very_complicated = tf.Variable([12.3 - 2.85j, 7.5 - 6.23j], tf.complex64)

**2 阶张量：**

-  至少包含一行和一列

.. code:: python

   mymat = tf.Variable([[7], 
                        [11]], tf.int16)

   myxor = tf.Variable([[False, True], 
                        [True, False]], tf.bool)

   linear_squares = tf.Variable([[4], 
                                 [9], 
                                 [16], 
                                 [25]], tf.int32)

   squarish_squares = tf.Variable([[4, 9], 
                                   [16, 25]], tf.int32)
   rank_of_squares = tf.rank(squarish_squares)


   mymatC = tf.Variable([[7], 
                         [11]], tf.int32)

**n 阶张量：**

-  更高阶的张量由一个n维数组组成，例如，在图像处理中，会使用4阶张量，维度分别对应批次大小、图像宽度、图像高度、颜色通道；

.. code:: python

   my_image = tf.zeros([10, 299, 299, 3])

**tf.Tensor 的阶 (rank)：**

-  ``tf.rank()``

**tf.Tensor 切片：**

-  由于tf.Tensor是n维单元数组，因此要访问tf.Tensor中的某一个单元，需要指定n个索引

.. code:: python

   my_scalar = my_vector[2]
   my_scalar = mymatrix[1, 2]
   my_row_vector = my_matrix[2]
   my_column_vector = my_matrix[:, 3]


2.1.2 形状(shape)
^^^^^^^^^^^^^^^^^^^^^

   -  张量的形状是每个维度中元素的数量

   -  TensorFlow
      文件编制中通过三种符号约定来描述张量维度：阶，形状和维数

   -  形状可以通过整型Python list, tuple, tf.TensorShape表示；

+----+---------------------+------+---------------------------------+
| 阶 | 形状                | 维数 | 示例                            |
+====+=====================+======+=================================+
| 0  | []                  | 0-D  | 0维张量，标量                   |
+----+---------------------+------+---------------------------------+
| 1  | [D0]                | 1-D  | 形状为[5]的1维张量              |
+----+---------------------+------+---------------------------------+
| 2  | [D0, D1]            | 2-D  | 形状为[3,4]的2维张量            |
+----+---------------------+------+---------------------------------+
| 3  | [D0, D1, D2]        | 3-D  | 形状为[1,4,3]的3维张量          |
+----+---------------------+------+---------------------------------+
| n  | [D0, D1, ..., Dn-1] | n维  | 形状为[D0,D1,...,Dn-1]的n维张量 |
+----+---------------------+------+---------------------------------+

**获取tf.Tensor对象的形状：**

-  方法：\ ``.shape`` => 返回 ``tf.TensorShape``

-  函数：\ ``tf.shape()``

.. code:: python

   zeros = tf.zeros(my_matrix.shape[1])

**改变tf.Tensor对象的形状：**

-  ``tf.reshape()``

.. code:: python

   rank_three_tensor = tf.ones([3, 4, 5])
   matrix = tf.reshape(rank_three_tensor, [6, 10])
   matrixB = tf.reshape(matrix, [3, -1])
   matrixAlt = tf.reshape(matrixB, [4, 3, -1])

2.1.3 数据类型
^^^^^^^^^^^^^^^^^^^^^

-  tf.DType：张量数据类型

   -  tf.as_dtype()

      -  convert numpy types and string type names to a ``tf.DType``
         object；

   -  int

      -  tf.int8

      -  tf.uint8

      -  tf.int16

      -  tf.uint16

      -  tf.int32

      -  tf.uint32

      -  tf.int64

      -  tf.uint64

      -  tf.qint8

      -  tf.quint8

      -  tf.qint16

      -  tf.quint16

      -  tf.qint32

   -  float

      -  tf.float16

      -  tf.float32

      -  tf.float64

      -  tf.bfloat16

   -  complex

      -  tf.complex64

      -  tf.complex128

   -  tf.string

   -  tf.bool

   -  tf.resource

   -  tf.variant

-  一个 ``tf.Tensor`` 只能有一种数据类型

   -  可以将任意数据结构序列化为string, 并将其存储在tf.Tensor中

-  tf.cast(): 将数据类型转换为另一种

-  可以使用方法 ``.dtype`` 检查tf.Tensor数据类型

-  用 python 对象创建 tf.Tensor
   时，可以选择指定数据类型。如果不指定数据类型，TensorFlow
   会选择一个可以表示您的数据的数据类型。TensorFlow 会将 Python
   整数转型为 tf.int32，并将 python 浮点数转型为
   tf.float32。此外，TensorFlow 使用 Numpy 在转换至数组时使用的相同规则


2.1.4 评估张量
^^^^^^^^^^^^^^^^^^^^^

   -  计算图构建完毕后，可以运行生成特定的tf.Tensor的计算并获取分配给它的值；

   -  .eval()方法仅在默认\ ``tf.Session``\ 处于活跃状态时才起作用；

   -  .eval()会返回一个与张量内容相同的Numpy数组；

.. code:: python

   constant = tf.constant([1, 2, 3])
   tensor = constant * constant
   print(tensor.eval())


2.1.5 输出张量
^^^^^^^^^^^^^^^^^^^^^

   出于调试目的，您可能需要输出 tf.Tensor 的值。虽然 tfdbg
   提供高级调试支持，但 TensorFlow
   也有一个操作\ ``tf.Print()``\ 可以直接输出 tf.Tensor 的值；

-  print()会输出tf.Tensor对象(表示延迟计算)，而不是其值；

.. code:: python

   # 输出tf.Tensor时很少使用一下模式`print()`
   t = <<some tensorflow operation>>
   print(t)

-  tf.Print()：返回其第一个张量参数(保持不变)，同时输出作为第二个参数传递的tf.Tensor集合；

.. code:: python

   t = <<some tensorflow operation>>
   tf.Print(t, [t])
   t = tf.Print(t, [t])
   result = t + 1



2.2 变量
~~~~~~~~

   -  TensorFlow变量是表示程序处理的共享持久状态的最佳方法；

      -  Tensorflow 使用 tf.Variable类操作张量；

      -  tf.Variable表示可通过对其运行操作来改变其值的张量；

      -  与tf.Tensor对象不同，tf.Variable存在于单个Session.run调用的上下文之外；

**APIs：**

   -  tf.get_variable()


2.2.1 创建变量
^^^^^^^^^^^^^^^^^^^^^

**tf.get_variable(name, shape)**

.. code:: python

   my_variable = tf.get_variable(name = "my_variable", 
                                 shape = [1, 2, 3])

   my_int_variable = tf.get_variable(name = "my_int_vaiable", 
                                     shape = [1, 2, 3],
                                     dtype = tf.int32,
                                     initializer = tf.zeros_initializer())

   other_variable = tf.get_variable(name = "other_variable", 
                                    dtype = tf.int32,
                                    initializer = tf.constant([32, 42]))

**变量集合**

   -  由于 TensorFlow
      程序的未连接部分可能需要创建变量，因此能有一种方式访问所有变量有时十分受用。为此，TensorFlow
      提供了集合，它们是张量或其他对象（如 tf.Variable
      实例）的命名列表；

   -  默认情况下，每个tf.Variable都放置在两个集合中：

      -  tf.GraphKeys.GLOBAL_VARIABLES

         -  可以在多台设备间共享的变量

      -  tf.GraphKeys.TRAINABLE_VARIABLES

         -  TensorFlow将计算其梯度的变量
            如果不希望变量可循量，可以将其添加到tf.GraphKeys.LOCAL_VARIABLES集合中

.. code:: python

   my_local = tf.get_variable(name = "my_local", 
                              shape = (),
                              collections = [tf.GraphKeys.LOCAL_VARIABLES])

   # or

   my_non_trainable = tf.get_variable(name = "my_non_trainable", 
                                      shape = (),
                                      trainable = False)

**变量集合**


2.2.2 初始化变量
^^^^^^^^^^^^^^^^^^^^^



2.2.3 使用变量
^^^^^^^^^^^^^^^^^^^^^

   要在TensorFlow图中使用tf.Variable的值，只需将其视为普通tf.Tensor即可；

.. code:: python

   v = tf.get_variable(name = "v", 
                       shape = (), 
                       initializer = tf.zeros_initializer())
   w = v + 1

.. _header-n385:

2.2.4 共享变量
^^^^^^^^^^^^^^^^^^^^^

   两种共享变量的方式：

   -  显式传递tf.Variable对象；

   -  将tf.Variable对象隐式封装在tf.variable_scope对象内；

.. _header-n394:

2.3 图和会话
~~~~~~~~~~~~

   -  可以将TensorFlow Core程序看作由两个互相独立的部分组成：

      -  1.构建计算图: tf.Graph

         -  计算图是排列成一个图的一系列TensorFlow指令，图由两种类型的对象组成：

            -  操作(op): 图的节点。描述了消耗和生成张量的计算；

            -  张量:
               图的边。代表将流经图的值。大多数TensorFlow函数会返回tf.Tensor

               -  打印张量并不会输出具体的对象值，只会构建计算图，tf.Tensor对象仅代表将要运行的操作结果，并且每个指令都有唯一的名称，后面跟着索引；

      -  2.运行计算图: tf.Session

         -  实例化一个tf.Session对象

         -  会话会封装TensorFlow运行的状态，并运行TensorFlow操作

         -  在调用tf.Session.run期间，任何tf.Tensor都只有单个值

**构建计算图(tf.Graph)：**

.. code:: python

   a = tf.constant(3.0, dtype = tf.float32)
   b = tf.constant(4.0)
   total = a + b
   print(a)
   print(b)
   print(total)

**运行计算图(tf.Session)：**

.. code:: python

   # 创建一个tf.Session对象的实例
   sess = tf.Session()
   print(sess.run(total))

传递多个张量给tf.Session.run:

.. code:: python

   print(sess.run({
       'ab': (a, b),
       'total': total
   }))

.. _header-n430:

2.4 保存和恢复
~~~~~~~~~~~~~~
