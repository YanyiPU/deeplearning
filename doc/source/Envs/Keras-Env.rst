
Keras Env
=============

Keras comes packaged with TensorFlow 2.0 as ``tensorflow.keras``. 
To start using Keras, simply install TensorFlow 2.0.

-  Keras/Tensorflow 兼容的设备

   -  Python 3.5-3.8

   -  Ubuntu 16.04 or later

   -  Windows 7 or later

   -  macOS 10.12.6(Sierra) or later

-  在安装 Keras 之前, 需要安装一个后端引擎: TensorFlow, Theano, CNTK

   -  `TensorFlow
      安装说明 <https://www.tensorflow.org/install>`__

   -  `Theano
      安装说明 <http://deeplearning.net/software/theano/install.html#install>`__

   -  `CNTK
      安装说明 <https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine>`__

-  安装可选依赖项

   -  cuDNN(如果您计划在GPU上运行Keras, 建议使用)

   -  HDF5和h5py(如果您计划将Keras模型保存到磁盘, 则需要)

   -  graphviz和pydot(由可视化实用程序用于绘制模型图)

-  安装 Keras

-  配置 Keras 后端


1. 安装后端引擎
----------------------

**Tensorflow 2.0:**

.. code:: shell

   # Tensorflow GPU version
   pip install --upgrade tensorflow-gpu

   # Tensorflow CPU version
   pip install --upgrade tensorflow

**Theano:**

.. code:: shell

   $ todo

**CNTK:**

.. code:: shell

   $ todo

2. 安装可选依赖项
----------------------

**cUNN:**

**HDF5,h5py:**

.. code:: shell

   $ conda install h5py

or

.. code:: shell

   $ pip install h5py

**graphviz:**

.. code:: shell

   $ pip install

**pydot:**

.. code:: shell

   $ pip install 

3. 安装 Keras
----------------------

**从 PYPI 安装 Keras:**

.. code:: shell

   # keras nonvirtualenv version
   $ sudo pip install keras

   # keras virtualenv version
   $ pip install keras

   # tensorflow version
   $ pip install keras -U --pre

**从 GitHub 源安装 Keras:**

.. code:: shell

   $ cd /usr/local
   $ git clone https://github.com/keras-team/keras.git
   $ cd keras
   $ sudo python setup.py install

4. 配置 Keras 后端
----------------------

默认情况下, Keras 将使用 Tensorflow 作为其张量操作库, 配置步骤如下：

`配置文档 <https://keras.io/backend/>`__


