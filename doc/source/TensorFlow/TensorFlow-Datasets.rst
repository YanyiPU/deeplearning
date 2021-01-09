
TensorFlow Datasets
=========================

1.TensorFlow Datasets 安装及使用
----------------------------------

   - 库安装:

      .. code-block:: shell

         $ pip install tensorflow
         $ pip install tensorflow-datasets

   - 库导入:

      .. code-block:: python

         # tf.data, tf.data.Dataset
         import tensorflow as tf
         # tf.keras.datasets.<dataset_name>.load_data
         from tensorflow.keras import datasets
         # tfds.load
         import tensorflow_datasets as tfds

2.TensorFlow Datasets 介绍
----------------------------------

2.1 数据集对象介绍
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - TensorFlow Datasets 是可用于 TensorFlow 或其他 Python 机器学习框架(例如 Jax) 的一系列数据集。
     所有数据集都作为 ``tf.data.Dataset`` 提供，实现易用且高性能的输入流水线。

   - TensorFlow 提供了 ``tf.data`` 模块，它包括了一套灵活的数据集构建 API，
     能够帮助快速、高效地构建数据输入的流水线，尤其适用于数据量巨大的情景

      - ``tf.data`` 的核心是 ``tf.data.Dataset`` 类，提供了对数据集的高层封装
      - ``tf.data.Dataset`` 由一系列可迭代访问的元素(element)组成，每个元素包含一个或多个张量
   
   - TensorFlow 数据集 API

      - ``tf.data``
         - ``tf.data.Dataset``
      - ``tensorflow_datasets``
      - ``tf.keras.datasets``

2.2 数据集对象的建立
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   1.建立 ``tf.data.Dataset`` 的最基本的方法是使用 ``tf.data.Dataset.from_tensor_slices()``

      - 使用与数据量较小(能够将数据全部装进内存)的情况
      - 如果数据集中的所有元素通过张量的第 0 维拼接成一个大的张量

      .. code-block:: python

         import tensorflow as tf
         import numpy as np

         X = tf.constant([2013, 2014, 2015, 2016, 2017])
         Y = tf.constant([12000, 14000, 15000, 16500, 17500])

         dataset = tf.data.Dataset.from_tensor_slices((X, Y))
         for x, y in dataset:
            print(x.numpy(), y.numpy())

   2.使用 ``tf.data.Dataset.from_tensor_slices()``、``tf.keras.datasets.mnist.load_data()``

      .. code-block:: python

         import tensorflow as tf
         import matplotlib.pyplot as plt

         (train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
         train_data = np.expand_dim(train_data.astype(np.float32) / 255, axis = -1)
         mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))

         for image, label in mnist_dataset:
            plt.title(label.numpy())
            plt.imshow(image.numpy()[:, :, 0])
            plt.show()

   3.TensorFlow Datasets 提供了一个基于 ``tf.data.Dataset`` 的开箱即用的数据集合

      .. code-block:: python

         import tensorflow_datasets as tfds

         # 构建 tf.data.Dataset
         dataset1 = tfds.load("mnist", split = "train", shuffle_files = True)
         dataset2 = tfds.load("mnist", split = tfds.Split.TRAIN, as_supervised = True)

         # 构建输入数据 Pipeline
         dataset1 = dataset1 \
            .shuffle(1024) \
            .batch(32) \
            .prefetch(tf.data.experimential.AUTOTUNE)
         
         for example in dataset1.take(1):
            image, label = example["image"], example["label"]

.. note:: 

   - 对于特别巨大而无法完整载入内存的数据集，可以先将数据集处理为 ``TFRecord`` 格式，
     然后使用 ``tf.data.TFRecordDataset()`` 进行载入

2.3 内置数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   每一个数据集都通过实现了抽象基类 ``tfds.core.DatasetBuilder`` 来构建.

1.查看数据集

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

2.数据集分类

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

2.4 获取内置数据集
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







3.TensorFlow Datasets
-------------------------------------------

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



3.3.1 导入数据
^^^^^^^^^^^^^^

   -  ``tf.data`` API 在 TensorFlow 中引入了两个新的抽象类：

      -  ``tf.data.Dataset``

         -  表示一系列元素，其中每个元素包含一个或多个 ``Tensor``
            对象。可以通过两种方式来创建数据集：

            -  ``tf.data.Dataset.from_tensor_slices()`` 通过一个或多个
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

3.1.2 数据集结构
^^^^^^^^^^^^^^^^^^^^^






4.TensorFlow Dataset 预处理数据
-------------------------------------

4.1 数据集预处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.1.1 数据集预处理 API 介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - ``tf.data.Dataset`` 类提供了多种数据集预处理方法:
      - ``tf.data.Dataset.map(f)``: 
         - 对数据集中的每个元素应用函数 ``f``，得到一个新的数据集
         - 结合 ``tf.io`` 对文件进行读写和解码
         - 结合 ``tf.image`` 进行图像处理
      - ``tf.data.Dataset.shuffle(buffer_size)``: 
         - 将数据集打乱
         - 设定一个固定大小的缓冲区(buffer)，取出前 buffer_size 个元素放入，并从缓冲区中随机采样，采样后的数据用后续数据替换
      - ``tf.data.Dataset.batch(batch_size)``: 
         - 将数据集分成批次
         - 对每 ``batch_size`` 个元素，使用 ``tf.stack()`` 在第 0 维合并，成为一个元素
      - ``tf.data.Dataset.repeat()``: 
         - 重复数据集的元素
      - ``tf.data.Dataset.reduce()``: 
         - 与 Map 相对的聚合操作
      - ``tf.data.Dataset.take()``: 
         - 截取数据集中的前若干个元素
   - ``tf.data.Dataset.prefetch()``:
      - 并行化策略提高训练流程效率
   - 获取与使用 ``tf.data.Dataset`` 数据集元素
      - ``tf.data.Dataset`` 是一个 Python 的可迭代对象

4.1.2 数据集处理示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- (1)使用 ``tf.data.Dataset.map()`` 将所有图片旋转 90 度

   .. code-block:: python
      
      import tensorflow as tf

      # data preprocessing function
      def rot90(image, label):
         image = tf.image.rot90(image)
         return image, label
      
      # data
      mnist_dataset = tf.keras.datasets.mnist.load_data()
      
      # data preprocessing
      mnist_dataset = mnist_dataset.map(rot90)

      # data visual
      for image, label in mnist_dataset:
         plt.title(label.numpy())
         plt.imshow(image.numpy()[:, :, 0])
         plt.show()

- (2)使用 ``tf.data.Dataset.batch()`` 将数据集划分为批次，每个批次的大小为 4

   .. code-block:: python

      import tensorflow as tf
      
      # data
      mnist_dataset = tf.keras.datasets.mnist.load_data()
      
      # data preprocessing
      mnist_dataset = mnist_dataset.batch(4)

      # data visual
      for images, labels in mnist_dataset: # image: [4, 28, 28, 1], labels: [4]
         fig, axs = plt.subplots(1, 4)
         for i in range(4):
            axs[i].set_title(label.numpy()[i])
            axs[i].imshow(images.numpy()[i, :, :, 0])
         plt.show()

- (3)使用 ``tf.data.Dataset.shuffle()`` 将数据打散后再设置批次，缓存大小设置为 10000

   .. code-block:: python

      import tensorflow as tf
      
      # data
      mnist_dataset = tf.keras.datasets.mnist.load_data()
      
      # data preprocessing
      mnist_dataset = mnist_dataset.shuffle(buffer_size = 10000).batch(4)

      # data visual
      for i in range(2):
         for images, labels in mnist_dataset: # image: [4, 28, 28, 1], labels: [4]
            fig, axs = plt.subplots(1, 4)
            for i in range(4):
               axs[i].set_title(label.numpy()[i])
               axs[i].imshow(images.numpy()[i, :, :, 0])
            plt.show()

   .. note:: 
   
      - 一般而言，若数据集的顺序分布较为随机，则缓冲区的大小可较小，否则需要设置较大的缓冲区

- (4)使用 ``tf.data.Dataset.prefetch()`` 并行化策略提高训练流程效率

   - 常规的训练流程
      - 当训练模型时，希望充分利用计算资源，减少 CPU/GPU 的空载时间，然而，有时数据集的准备处理非常耗时，
         使得在每进行一次训练前都需要花费大量的时间准备带训练的数据，GPU 只能空载等待数据，造成了计算资源的浪费

   - 使用 ``tf.data.Dataset.prefetch()`` 方法进行数据预加载后的训练流程
      - ``tf.data.Dataset.prefetch()`` 可以让数据集对象 ``Dataset`` 在训练时预先取出若干个元素，
         使得在 GPU 训练的同时 CPU 可以准备数据，从而提升训练流程的效率

   .. code-block:: python

      import tensorflow as tf
      
      # data preprocessing function
      def rot90(image, label):
         image = tf.image.rot90(image)
         return image, label
      
      # data
      mnist_dataset = tf.keras.datasets.mnist.load_data()
      
      # data preprocessing
      # 开启数据预加载功能
      mnist_dataset = mnist_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
      # 利用多 GPU 资源，并行化地对数据进行变换
      mnist_dataset = mnist_dataset.map(map_func = rot90, num_parallel_calls = 2)
      mnist_dataset = mnist_dataset.map(map_func = rot90, num_parallel_calls = tf.data.experimental.AUTOTUNE)

- (5)获取与使用 ``tf.data.Dataset`` 数据集元素

   - 构建好数据并预处理后，需要从中迭代获取数据用于训练

   .. code-block:: python

      dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
      for a, b, c ... in dataset:
         pass

   .. code-block:: python

      dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
      it = iter(dataset)
      a_0, b_0, c_0, ... = next(it)
      a_1, b_1, c_1, ... = next(it)

4.2 图像
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.2 文本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.3 CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.4 Numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.5 pandas.DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.6 Unicode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.7 TF.Text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.8 TFRecord
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.8.1 TFRecord 数据文件介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   TFRecord 是 TensorFlow 中的数据集存储格式。当将数据集整理成 TFRecord 格式后，
   TensorFlow 就可以高效地读取和处理这些数据集了。从而帮助更高效地进行大规模模型训练。

   TFRecord 可以理解为一系列序列化的 ``tf.train.Example`` 元素所组成的列表文件，
   而每一个 ``tf.train.Example`` 又由若干个 ``tf.train.Feature`` 的字典组成：

      .. code-block:: python
      
         # dataset.tfrecords
         [
            {  # example 1 (tf.train.Example)
               'feature_1': tf.train.Feature,
               ...
               'feature_k': tf.train.Feature,
            },
            ...
            {  # example N (tf.train.Example)
               'feature_1': tf.train.Feature,
               ...
               'feature_k': tf.train.Feature,
            }, 
         ]

4.8.2 TFRecord 文件保存
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- TFRecord 文件保存步骤

   为了将形式各样的数据集整理为 TFRecord 格式，可以对数据集中的每个元素进行以下步骤：

      - (1) 读取该数据元素到内存
      - (2) 将该元素转换为 ``tf.train.Example`` 对象

         - 每个 ``tf.train.Example`` 对象由若干个 ``tf.train.Feature`` 的字典组成，因此需要先建立 Feature 的子典

      - (3) 将 ``tf.train.Example`` 对象序列化为字符串，并通过一个预先定义的 ``tf.io.TFRecordWriter`` 写入 ``TFRecord`` 文件

- TFRecord 文件保存示例

   .. code-block:: python

      import tensorflow as tf
      import os

      # root
      root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
      # project
      project_path = os.path.join(root_dir, "deeplearning/src/tensorflow_src")
      # model save
      models_path = os.path.join(project_path, "save")
      # data
      cats_and_dogs_dir = os.path.join(root_dir, "datasets/cats_vs_dogs")
      data_dir = os.path.join(root_dir, "datasets/cats_vs_dogs/cats_and_dogs_small")
      # train data
      train_dir = os.path.join(data_dir, "train")
      train_cats_dir = os.path.join(train_dir, "cat")
      train_dogs_dir = os.path.join(train_dir, "dog")
      # tfrecord
      tfrecord_file = os.path.join(cats_and_dogs_dir, "train.tfrecord")

      # 训练数据
      train_cat_filenames = [os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)]
      train_dog_filenames = [os.path.join(train_dogs_dir, filename) for filename in os.listdir(train_dogs_dir)]
      train_filenames = train_cat_filenames + train_dog_filenames
      train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

      # 迭代读取每张图片，建立 tf.train.Feature 字典和 tf.train.Example 对象，序列化并写入 TFRecord
      with tf.io.TFRecordWriter(tfrecord_file) as writer:
         for filename, label in zip(train_filenames, train_labels):
            # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
            image = open(filename, "rb").read()
            # 建立 tf.train.Feature 字典
            feature = {
                  # 图片是一个 Byte 对象
                  "image": tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                  "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
            }
            # 通过字典建立 Example
            example = tf.train.Example(features = tf.train.Features(feature = feature))
            # 将 Example 序列化并写入 TFRecord 文件
            writer.write(example.SerializeToString())



4.8.3 TFRecord 文件读取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- TFRecord 数据文件读取步骤

      - (1)通过 ``tf.data.TFRecordDataset`` 读入原始的 TFRecord 文件，获得一个 ``tf.data.Dataset`` 数据集对象

         - 此时文件中的 ``tf.train.Example`` 对象尚未被反序列化

      - (2)通过 ``tf.data.Dataset.map`` 方法，对该数据集对象中的每个序列化的 ``tf.train.Example`` 字符串
        执行 ``tf.io.parse_single_example`` 函数，从而实现反序列化

- TFRecord 数据文件读取示例

   .. code-block:: python

      import tensorflow as tf
      import os
      import matplotlib.pyplot as plt

      # root
      root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
      # data
      cats_and_dogs_dir = os.path.join(root_dir, "datasets/cats_vs_dogs")
      # tfrecord
      tfrecord_file = os.path.join(cats_and_dogs_dir, "train.tfrecord")

      def _parse_example(example_string):
         """
         将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
         """
         # 定义 Feature 结构，告诉解码器每个 Feature 的类型是什么
         feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)
         }
         feature_dict = tf.io.parse_single_example(example_string, feature_description)
         # 解码 JPEG 图片
         feature_dict["image"] = tf.io.decode_jpeg(feature_dict["image"])
         return feature_dict["image"], feature_dict["label"]

      # 读取 TFRecord 文件
      raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
      dataset = raw_dataset.map(_parse_example)

      for image, label in dataset:
         plt.title("cat" if label == 0 else "dog")
         plt.imshow(image.numpy())
         plt.show()

4.9 tf.io 的其他格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



4.10 tf.TensorArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.10.1 tf.TensorArray 介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   在部分网络结构中，尤其是涉及时间序列的结构中，可能需要将一系列张量以数组的方式依次存放起来，以供进一步处理。

      - 在即时执行模式下，可以直接使用一个 Python 列表存放数组
      - 如果需要基于计算图的特性，例如使用 @tf.function 加速模型运行或者使用 SaveModel 导出模型，就无法使用 Python 列表了

   TensorFlow 提供了 ``tf.TensorArray`` (TensorFlow 动态数组) 支持计算图特性的 TensorFlow 动态数组.

      - 声明方式如下:

         - ``arr = tf.TensorArray(dtype, size, dynamic_size = False)``: 

            - 声明一个大小为 ``size``，类型为 ``dtype`` 的 ``TensorArray arr``
            - 如果将 ``dynamic_size`` 参数设置为 True，则该数组会自动增长空间

      - 读取和写入的方法如下:

         - ``write(index, value)``: 将 value 写入数组的第 index 个位置
         - ``read(index)``: 读取数组的第 index 个值
         - ``stack()``
         - ``unstack()``

4.10.2 tf.TensorArray 介绍
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. code-block:: python

      import tensorflow as tf

      @tf.function
      def array_write_and_read():
         arr = tf.TensorArray(dtype = tf.float32, size = 3)
         arr = arr.write(0, tf.constant(0.0))
         arr = arr.write(1, tf.constant(1.0))
         arr = arr.write(2, tf.constant(2.0))
         arr_0 = arr.read(0)
         arr_1 = arr.read(1)
         arr_2 = arr.read(2)
         return arr_0, arr_1, arr_2
      
      a, b, c = array_write_and_read()
      print(a, b, c)

.. note:: 

   - 由于需要支持计算图，``tf.TensorArray`` 的 ``write()`` 是不可以忽略左值的，
     也就是说，在图执行模式下，必须按照以下的形式写入数组，才可以正常生成一个计算图操作，
     并将该操作返回给 ``arr``:

      .. code-block:: python

         arr.write(index, value)

   - 不可以写成

      .. code-block:: python

         arr.write(index, value)




5.数据输入流水线
---------------------

5.1 tf.data
~~~~~~~~~~~~~~~~~~~

5.2 优化流水线性能
~~~~~~~~~~~~~~~~~~~~

5.3 分析流水线性能
~~~~~~~~~~~~~~~~~~~~~~

