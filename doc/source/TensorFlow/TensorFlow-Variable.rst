.. _header-n0:

TensorFlow Variable
======================

.. _header-n3:

APIs
----

-  class ``tf.Variable``

   -  create

   -  update

   -  manage

   -  `Doc <https://tensorflow.google.cn/api_docs/python/tf/Variable>`__

.. _header-n18:

创建变量、初始化变量
--------------------

.. code:: python

   my_variable = tf.Variable(tf.zeros([1., 2., 3.]))

.. code:: python

   with tf.device("/device:GPU:1"):
   	"""
   	# If there's a tf.device scope active, 
   	# the variable will be placed on that device.
   	# otherwise the variable will be placed on 
   	# the "fastest" device compatible with its dtype
   	"""
   	v = tf.Variable(tf.zeros([10, 10]))

.. _header-n21:

使用变量
--------

-  在 TensorFlow 图中使用一个 ``tf.Variable`` 时，只需将这个变量当做一个
   ``tf.Tensor``\ 即可

.. code:: python

   v = tf.Variable(0.0)
   w = v + 1

.. code:: python

   v = tf.Variable(0.0)
   v.assign_add(1)
   v.read_value()

.. _header-n28:

Keep track of variables
-----------------------

-  TensorFlow 中的一个变量是一个 Python 对象.

-  可以列出一个模型中的所有变量.

.. code:: python

   class MyLayer(tf.keras.layers.Layer):

   	def __init__(self):
   		super(MyLayer, self).__init__()
   		self.my_var = tf.Variable(1.0)
   		self.my_var_list = [tf.Variable(x) for x in range(10)]


   class MyOtherLayer(tf.keras.layers.Layer):

   	def __init__(self):
   		super(MyOtherLayer, self).__init__()
   		self.sublayer = MyLayer()
   		self.my_other_var = tf.Variable(10.0)

   m = MyOtherLayer()
   print(len(m.variables)) # 12
