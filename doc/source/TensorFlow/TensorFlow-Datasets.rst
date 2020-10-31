.. _header-n0:

TensorFlow Datasets
=====================

.. _header-n108:

1.TensorFlow Datasets 安装及使用
----------------------------------

库安装:

   .. code-block:: shell

      $ pip install tensorflow
      $ pip install tensorflow-datasets

库导入:

   .. code-block:: python
   
      import numpy as np
      import tensorflow as tf
      import matplotlib.pyplot as plt
      import tensorflow_datasets as tfds


.. _header-n110:

2.TensorFlow Datasets 介绍
----------------------------------

2.1 介绍
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow Datasets 是可用于 TensorFlow 或其他 Python 机器学习框架(例如 Jax) 的一系列数据集。
所有数据集都作为 ``tf.data.Datasets`` 提供，实现易用且高性能的输入流水线。


示例:

   .. code-block:: python

      import tensorflow as tf
      import tensorflow_datasets as tfds

      # Construct a tf.data.Dataset
      ds =  tfds.load("mnist", split = "train", shuffle_files = True)

      # Build your input pipeline
      ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimential.AUTOTUNE)
      for example in ds.take(1):
         image, label = example["image"], example["label"]


2.2 内置数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每一个数据集都通过实现了抽象基类 ``tfds.core.DatasetBuilder`` 来构建.

查看数据集：

.. code-block:: python

   import tensorflow as tf
   import tensorflow_datasets as tfds

   # 所有可用的数据集
   print(tfds.list_builders()) 

查看数据集结果：

.. code-block:: 

   ['abstract_reasoning', 'aflw2k3d', 'amazon_us_reviews', 
    'bair_robot_pushing_small', 'bigearthnet', 'binarized_mnist', 'binary_alpha_digits', 
    'caltech101', 'caltech_birds2010', 'caltech_birds2011', 'cats_vs_dogs', 'celeb_a', 'celeb_a_hq', 'chexpert', 'cifar10', 'cifar100', 'cifar10_corrupted', 'clevr', 'cnn_dailymail', 'coco', 'coco2014', 'coil100', 'colorectal_histology', 'colorectal_histology_large', 'curated_breast_imaging_ddsm', 'cycle_gan', 
    'deep_weeds', 'definite_pronoun_resolution', 'diabetic_retinopathy_detection', 'downsampled_imagenet', 'dsprites', 'dtd', 'dummy_dataset_shared_generator', 'dummy_mnist', 
    'emnist', 'eurosat', 
    'fashion_mnist', 'flores', 'food101', 
    'gap', 'glue', 'groove', 
    'higgs', 'horses_or_humans', 
    'image_label_folder', 'imagenet2012', 'imagenet2012_corrupted', 'imdb_reviews', 'iris', 'kitti', 
    'kmnist', 
    'lfw', 'lm1b', 'lsun', 
    'mnist', 'mnist_corrupted', 'moving_mnist', 'multi_nli', 
    'nsynth', 
    'omniglot', 'open_images_v4', 'oxford_flowers102', 'oxford_iiit_pet', 
    'para_crawl', 'patch_camelyon', 'pet_finder', 'quickdraw_bitmap', 
    'resisc45', 'rock_paper_scissors', 'rock_you', 
    'scene_parse150', 'shapes3d', 'smallnorb', 'snli', 'so2sat', 'squad', 'stanford_dogs', 'stanford_online_products', 'starcraft_video', 'sun397', 'super_glue', 'svhn_cropped', 
    'ted_hrlr_translate', 'ted_multi_translate', 'tf_flowers', 'titanic', 'trivia_qa', 
    'uc_merced', 'ucf101', 
    'visual_domain_decathlon', 'voc2007', 
    'wikipedia', 'wmt14_translate', 'wmt15_translate', 'wmt16_translate', 'wmt17_translate', 'wmt18_translate', 'wmt19_translate', 'wmt_t2t_translate', 'wmt_translate', 
    'xnli']

数据集分类：

   -  Audio

      -  groove

      -  nsynth

   -  Image

      -  abstract_reasoning

      -  aflw2k3d

      -  bigearthnet

      -  binarized_mnist

      -  binary\ *alpha*\ digits

      -  caltech101

      -  caltech_birds2010

      -  caltech_birds2011

      -  cats\ *vs*\ dogs

      -  celeb_a

      -  celeb\ *a*\ hq

      -  cifar10

      -  cifar100

      -  cifar10_corrupted

      -  clevr

      -  coco

      -  coco2014

      -  coil100

      -  colorectal_histology

      -  colorectal\ *histology*\ large

      -  curated\ *breast*\ imaging_ddsm

      -  cycle_gan

      -  deep_weeds

      -  diabetic\ *retinopathy*\ detection

      -  downsampled_imagenet

      -  dsprites

      -  dtd

      -  emnist

      -  eurosat

      -  fashion_mnist

      -  food101

      -  horses\ *or*\ humans

      -  image\ *label*\ folder

      -  imagenet2012

      -  imagenet2012_corrupted

      -  kitti

      -  kmnist

      -  lfw

      -  lsun

      -  mnist

      -  mnist_corrupted

      -  omniglot

      -  open\ *images*\ v4

      -  oxford_flowers102

      -  oxford\ *iiit*\ pet

      -  patch_camelyon

      -  pet_finder

      -  quickdraw_bitmap

      -  resisc45

      -  rock\ *paper*\ scissors

      -  scene_parse150

      -  shapes3d

      -  smallnorb

      -  so2sat

      -  stanford_dogs

      -  stanford\ *online*\ products

      -  sun397

      -  svhn_cropped

      -  tf_flowers

      -  uc_merced

      -  visual\ *domain*\ decathlon

      -  voc2007

   -  Structured

      -  amazon\ *us*\ reviews

      -  higgs

      -  iris

      -  rock_you

      -  titanic

   -  Text

      -  cnn_dailymail

      -  definite\ *pronoun*\ resolution

      -  gap

      -  glue

      -  imdb_reviews

      -  lm1b

      -  multi_nli

      -  snli

      -  squad

      -  super_glue

      -  trivia_qa

      -  wikipedia

      -  xnli

   -  Translate

      -  flores

      -  para_crawl

      -  ted\ *hrlr*\ translate

      -  ted\ *multi*\ translate

      -  wmt14_translate

      -  wmt15_translate

      -  wmt16_translate

      -  wmt17_translate

      -  wmt18_translate

      -  wmt19_translate

      -  wmt\ *t2t*\ translate

   -  Video

      -  bair\ *robot*\ pushing_small

      -  moving_mnist

      -  starcraft_video

      -  ucf101

2.3 获取内置数据集
~~~~~~~~~~~~~~~~~~~~

``tfds.load`` 是构建并加载 ``tf.data.Dataset`` 最简单的方式。``tf.data.Dataset`` 是构建输入流水线的标准 TensorFlow 接口。

示例:

   .. code-block:: python

      mnist_train = tfds.load("mnist", split = "train", download = False, data_dir = "~/.tensorflow_datasets/")
      assert isinstance(mnist_train, tf.data.Dataset)
      print(mnist_train)


2.4 特征字典
~~~~~~~~~~~~~~~~~~~~~

所有 ``tfds`` 数据集都包含将特征名称映射到 Tensor 值的特征字典。典型的数据集将具有 2 个键:

   - ``"image"``

   - ``"label"``

示例:

   .. code-block:: python

      mnist_train = tfds.load("mnist", split = "train", download = False, data_dir = "~/.tensorflow_datasets/")
      for mnist_example in mnist_train.take(1): # 只取一个样本
         image, label = mnist_example["image"], mnist_example["label"]
         plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cma = plt.get_cmap("gray"))
         print("Label: %d" % label.numpy())

2.5 DatasetBuilder
~~~~~~~~~~~~~~~~~~~~~~

``tfds.load`` 实际上是一个基于 ``DatasetBuilder`` 的简单方便的包装器

示例:

   .. code-block:: python

      mnist_builder = tfds.builder("mnist")
      mnsit_builder.download_and_prepare()
      mnist_train = mnist_builder.as_dataset(split = "train")
      mnist_train



2.6 输入流水线
~~~~~~~~~~~~~~~~~~~~~~~~

一旦有了 ``tf.data.Dataset`` 对象，就可以使用 ``tf.data`` 接口定义适合模型训练的输入流水线的其余部分.

示例:

   .. code-block:: python

      mnist_train = mnist_train.repeat().shuffle(1024).batch(32)

      # prefetch 将使输入流水线可以在模型训练时一步获取批处理
      mnist_train = mnist_train \
                     .repeat() \
                     .shuffle(1024) \
                     .batch(32) \
                     .prefetch(tf.data.experimental.AUTOTUNE)

2.7 数据集信息
~~~~~~~~~~~~~~~~~~~~~~~~

示例:

   .. code-block:: python

      # method 1
      info = mnist_builder.info
      print(info)
      print(info.features)
      print(info.features["label"].num_classes)
      print(info.features["label"].names)

      # method 2
      mnist_test, info = tfds.load("mnist", split = "test", with_info = True)
      print(info)


2.8 数据集可视化
~~~~~~~~~~~~~~~~~~~~~~~~

示例:

   .. code-block:: python

      fig = tfds.show_examples(info, mnist_test)






.. _header-n329:

3.TensorFlow Datasets
---------------------

.. _header-n330:

3.1 数据集的信息
~~~~~~~~~~~~~~~~

-  URL:

   -  \`\`

-  DatasetBuilder:

   -  ``tfds.structured.data.Data``

-  Version:

   -  ``v0.0.0``

-  Size:

   -  ``0.00 KiB/MiB``

-  Features:

   -  ``FeaturesDict({"": , "": })``

-  Statistics:

   -  Split

   -  TRAIN

   -  TEST

   -  ALL

-  Urls:

   -  \`\`

-  Citation:

   -  ``@misc{}``

.. _header-n378:

3.2 数据集的使用
~~~~~~~~~~~~~~~~

.. code:: python

   import tensorflow as tf
   import tensorflow_datasets as tfds

(1) 创建 ``tf.data.Dataset``:

.. code:: python

   # method 1
   mnist_data, info = tfds \
       .load(name = "mnist", 
             split = None,
             data_dir = "/Users/zfwang/data/tensorflow_datasets/",
             download = True,
             with_info = True)

   # method 2
   mnist_builder = tfds.builder("mnist")
   mnist_builder.download_and_prepare()
   mnist_data = mnist_builder.as_dataset(split = tfds.Split.TRAIN)
   info = mnist_builder.info

   # Numpy arrays
   np_mnist_data = tfds.as_numpy(mnist_data)

   print(mnist_data)
   print(np_mnist_data)
   print(info)

(2) Feature dictionaries

.. code:: python

   # feature dict
   for features in mnist_data.take(1):
       image, label = features["image"], features["label"]
   # or 
   for features in tfds.as_numpy(mnist_data):
       image, label = features["image"], features["label"]

   plt.imshow(image.numpy()[:, :, 0].astype(np.float32), 
              cmap = plt.get_cmap("gray"))
   print("Label: %d" % label.numpy())

(2) 创建 input pipeline:

-  一旦有了tf.data.Dataset对象，就可以使用
   `tf.data <https://www.tensorflow.org/guide/datasets>`__\ API
   定义适合模型训练的输入管道的其余部分

.. code:: python

   mnist_data = mnist_data \
       .shuffle(1024) \
       .batch(128) \
       .repeat(5) \
       .prefetch(tf.data.experimental.AUTOTUNE)


   print(mnist_data)
   print(image)
   print(label)

(3) 训练数据、测试数据：

.. code:: python

   mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]

.. code:: python

   mnist_train, mnist_test = tfds \
       .load(name = "mnist", 
             split = ["train", "test"],
             with_info = True)
   mnist_train = mnist_train \
       .shuffle(1000) \
       .batch(128) \
       .repeat(5) \
       .prefetch(10)
   for features in mnist_train.take(1):
       image, label = features["image"], features["label"]

.. _header-n393:

3.3 TensorFlow Datasets APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Modules

   -  core

   -  decode

   -  download

   -  features

   -  file_adapter

   -  testing

   -  units

-  Classes

   -  tfds.GenerateMode

   -  tfds.Split

      -  ``tfds.Split.TRAIN``

      -  ``tfds.Split.TEST``

      -  ``tfds.Split.VALIDATION``

      -  ``tfds.Split.ALL``

   -  tfds.percent

-  Functions

   -  ``tfds.list_builders()``

   -  ``tfds.load()``

   -  ``tfds.builder()``

   -  ``tfds.as_numpy(dataset, graph = None)``

   -  disable\ *progress*\ bar()

   -  is\ *dataset*\ on_gcs

   -  show_examples()

.. _header-n448:

(1) tfds.load()
^^^^^^^^^^^^^^^

-  ``tfds.load()``

-  ``tfds.builder()``

-  ``tfds.as_numpy()``

..

   -  Loads the named dataset into a ``tf.data.Dataset``

   -  ``tfds.core.DatasetBuilder``\ 的简单形式

      -  DatasetBuilder.download\ *and*\ prepare

      -  DatasetBuilder.as_dataset

.. code:: python

   tfds.load(
       name, # "mnist"
       split = None, 
       data_dir = None,
       batch_size = None,
       in_memory = None,
       shuffle_files = None,
       download = True,
       as_supervised = False,
       decoders = None,
       with_info = False,
       builder_kwargs = None,
       download_and_prepare_kwargs = None,
       as_dataset_kwargs = None,
       try_gcs = False
   )

   # euqal
   def tfds_load():
       builder = tfds.builder(name, data_dir = datadir, **builder_kwargs)
       if download:
           builder.download_and_prepare(**download_and_prepare_kwargs)
       ds = builder.as_dataset(split = split, as_supervised = as_supervised, **as_dataset_kwargs)
       if with_info:
           return ds, builder.info
       return ds


   # tf.data.Dataset or tf.Tensor => Numpy array
   tfds.as_numpy()

参数：

-  ``split``

   -  split = None

      -  return a dict with all splits

   -  split = "train"

   -  split = ["train", "test"]

   -  split = tfds.Split.TRAIN/TEST/VALIDATION/ALL

-  ``data_dir``

   -  "/User/zfwang/data/tensorflow_datasest/"

-  ``download``

   -  True

   -  False

-  with_info

   -  True

   -  False

返回值：

-  ds

   -  ``tf.data.Dataset``

   -  if ``split = None``

      -  dict ``<key: tfds.Split, value: tfds.data.Dataset>``

   -  if ``batch_size = -1``

      -  full datasets as ``tf.Tensor``

-  ds.info

.. _header-n523:

3.3.1 导入数据
^^^^^^^^^^^^^^

   -  ``tf.data`` API 在 TensorFlow 中引入了两个新的抽象类：

      -  ``tf.data.Dataset``

         -  表示一系列元素，其中每个元素包含一个或多个 ``Tensor``
            对象。可以通过两种方式来创建数据集：

            -  ``tf.data.Dataset.from_tensor_slice()`` 通过一个或多个
               ``tf.Tensor`` 对象创建数据集

            -  ``tf.data.Dataset.batch()`` 通过一个或多个
               ``tf.data.Dataset`` 对象创建数据集

      -  ``tf.data.Iterator``

         -  提供了从数据集中提取元素的主要方法。\ ``Iterator.get_nex()``
            返回的操作会在执行时生成 ``Dataset``
            的下一个元素，并且此操作通常充当管道代码和模型之间的接口。

**基本机制：**

**读取输入数据：**

**使用 Dataset.map() 预处理数据：**

**批处理数据集元素：**

**训练工作流程：**

   处理多个周期

-  ``tf.data`` API 提供了两种主要方式来处理统一数据的多个周期

   -  要迭代数据集多个周期，最简单的方法是使用 Dataset.repeat()
      转换。例如，要创建一个将其输入重复 10 个周期的数据集

   -  如果您想在每个周期结束时收到信号，则可以编写在数据集结束时捕获
      tf.errors.OutOfRangeError
      的训练循环。此时，您可以收集关于该周期的一些统计信息（例如验证错误）

method 1:

.. code:: python

   # 10 epochs, batch_size = 32
   filenames = ["/var/data/file1.tfrecord",
                "/var/data/file2.tfrecord"]
   dataset = tf.data.TFRecordDataset(filenames)
   dataset = dataset.map(...)
   dataset = dataset.repeat(10)
   dataset = dataset.batch(32)

method 2:

.. code:: python

   filenames = ["/var/data/file1.tfrecord",
                "/var/data/file2.tfrecord"]
   dataset = tf.data.TFRecordDataset(filenames)
   dataset = dataset.map(...)
   dataset = dataset.batch(32)
   iterator = dataset.make_initializable_iterator()
   next_element = iterator.get_next()

   # computer for 100 epochs
   for _ in range(100):
       sess.run(iterator.initializer)
       while True:
           try:
               sess.run(next_element)
           except tf.errors.OutOfRangeError:
               break


   随机重排输入数据

   使用高阶 API



3.4 导入数据
~~~~~~~~~~~~~~~~~~~~~~~

   -  API: ``tf.data``

      -  根据简单的可重用片段构建复杂的输入管道；

         -  图片模型：

         -  文本模型：

   -  tf.data在TensorFlow中引入两个新的抽象类：

      -  ``tf.data.Dataset``:
         表示一系列元素，其中每个元素包含一个或多个Tensor对象；

         -  两种创建数据集的方式：

            -  **创建来源**\ ：通过一个或多个tf.Tensor对象构建数据集

               -  ``tf.data.Dataset.from_tensors()``

               -  ``tf.data.Dataset.from_tensor_slices()``

            -  **应用转换**\ ：通过一个或多个tf.data.Dataset对象构建数据集

               -  ``tf.data.Dataset.map()``

               -  ``tf.data.Dataset.batch()``

      -  ``tf.data.Iterator``: 提供了从数据集中提取元素的主要方法

         -  ``tf.data.Iterator.get_next()``

            -  返回的操作会在执行时生成Dataset的下一个元素，并且此操作通常当输入管道和模型之间的接口

         -  ``tf.data.Iterator.initializer``

            -  使用不同的数据集重新初始化和参数化迭代器

.. _header-n53:

3.1.1 基本机制
^^^^^^^^^^^^^^^^^^^^^

   创建不同种类的Dataset和Iterator对象的基础知识，以及如何从这些对象中提取数据；

1. **定义数据来源——Dataset：**

   -  通过内存中的张量构建Dataset

      -  tf.data.Dataset.from_tensors()

      -  tf.data.Dataset.from\ *tensor*\ slices()

   -  通过以TFRecord格式存储在磁盘上的数据构建Dataset

      -  tf.data.TFRecordDataset

2. **将Dataset进行数据转换：**

   -  tf.data.Dataset的数据转换方法

      -  tf.data.Dataset.map()

         -  单元素转换，为每个元素应用一个函数

      -  tf.data.Dataset.batch()

         -  多元素转换

      -  ...

3. **从数据集中提取元素：**

   -  构建迭代器对象：

      -  tf.data.Iterator.initializer

         -  重新初始化迭代器的状态

      -  tf.data.Iterator.get_next()

         -  返回对应于有符号下一个元素的tf.Tensor对象

.. _header-n106:

3.1.2 数据集结构
^^^^^^^^^^^^^^^^^^^^^



4.加载和预处理数据
----------------------------


4.1 tf.data 数据集的构建与预处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow 提供了 ``tf.data`` 模块，它包含了一套灵活的数据集构建 API，能够帮助我们快速、高效地构建数据输入的流水线，
尤其适用于数据量巨大的场景。

4.1.1 数据集对象的建立
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``tf.data`` 的核心是 ``tf.data.Dataset`` 类，提供对数据集的高层封装。

``tf.data.Dataset`` 由一系列可迭代访问的元素(element)组成，每个元素包含一个或多个张量。





4.2 图像
~~~~~~~~~~~~~~~~~~~~~



4.2 文本
~~~~~~~~~~~~~~~~~~~~~



4.3 CSV
~~~~~~~~~~~~~~~~~~~~~


4.4 Numpy
~~~~~~~~~~~~~~~~~~~~~


4.5 pandas.DataFrame
~~~~~~~~~~~~~~~~~~~~~


4.6 Unicode
~~~~~~~~~~~~~~~~~~~~~


4.7 TF.Text
~~~~~~~~~~~~~~~~~~~~~


4.8 TFRecord 和 tf.Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.9 tf.io 的其他格式
~~~~~~~~~~~~~~~~~~~~~


5.数据输入流水线
---------------------

5.1 tf.data
~~~~~~~~~~~~~~~~~~~


5.2 优化流水线性能
~~~~~~~~~~~~~~~~~~~~



5.3 分析流水线性能
~~~~~~~~~~~~~~~~~~~~~~


