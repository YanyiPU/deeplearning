.. _header-n0:

TensorFlow EagerExecution
===========================

.. _header-n3:

TensorFlow Eager Execution
--------------------------

.. _header-n4:

1.设置
~~~~~~

升级到最新版本的TensorFlow：

.. code:: shell

   pip install -q --upgrade tensroflow==1.11

**启动Eager Execution：**

.. code:: python

   from __future__ import absolute_import, division, print_function
   import tensorflow as tf

   print(tf.executing_eagerly())
