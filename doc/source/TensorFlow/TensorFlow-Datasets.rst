
TensorFlow Datasets
=================================

1.TensorFlow Datasets 库
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

2.TensorFlow Dataset 介绍
----------------------------------

2.1 TensorFlow Dataset 介绍
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - TensorFlow 提供了 ``tf.data`` 模块，它包括了一套灵活的数据集构建 API，
     能够帮助快速、高效地构建数据输入的流水线，尤其适用于数据量巨大的情景

      - ``tf.data`` API 在 TensorFlow 中引入了两个新的抽象类：

         - ``tf.data.Dataset``: 提供了对数据集的高层封装

            - ``tf.data.Dataset`` 由一系列可迭代访问的元素(element)组成，其中每个元素包含一个或多个 ``Tensor`` 对象
            - ``tf.data.Dataset`` 可以通过两种方式来创建数据集：

               -  **创建来源**：通过一个或多个 ``tf.Tensor`` 对象构建数据集

                  -  ``tf.data.Dataset.from_tensors()``
                  -  ``tf.data.Dataset.from_tensor_slices()``

               -  **应用转换**：通过一个或多个 ``tf.data.Dataset`` 对象构建数据集

                  -  ``tf.data.Dataset.map()``
                  -  ``tf.data.Dataset.batch()``

         - ``tf.data.Iterator``

            - 提供了从数据集中提取元素的主要方法
            - ``tf.data.Iterator.get_next()``

               -  返回的操作会在执行时生成Dataset的下一个元素，并且此操作通常当输入管道和模型之间的接口

            - ``tf.data.Iterator.initializer``

               - 使用不同的数据集重新初始化和参数化迭代器

   - TensorFlow Datasets(``tensorflow_datasets``) 是可用于 TensorFlow 或其他 Python 机器学习框架(例如 Jax) 的一系列数据集。
     所有数据集都作为 ``tf.data.Dataset`` 提供，实现易用且高性能的输入流水线。
   - TensorFlow 数据集 API

      - ``tf.data``

         - ``tf.data.Dataset``
         - ``tf.data.Dataset.from_tensor_slices``

      - ``tensorflow_datasets``

         - ``tensorflow_datasets.load(data, split, shuffle_files, as_supervised)``

      - ``tf.keras.datasets``

         - ``tf.keras.datasets.mnist.load_data()``

2.2 TensorFlow Dataset 建立
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   1.建立 ``tf.data.Dataset`` 的最基本的方法是使用 ``tf.data.Dataset.from_tensor_slices()``

      - 适用于数据量较小(能够将数据全部装进内存)的情况
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

         for image, label in mnist_dataset.take(1):
            plt.title(label.numpy())
            plt.imshow(image.numpy())
            plt.show()

   3.TensorFlow Datasets 提供了一系列可以和 Tensorflow 配合使用的数据集，它负责下载和准备数据，以及构建 ``tf.data.Dataset``

      - 每一个数据集(dataset) 都实现了抽象基类 tfds.core.DatasetBuilder 来构建

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

2.3 TensorFlow 内置 Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - TensorFlow Datasets 提供了一系列可以和 Tensorflow 配合使用的数据集，它负责下载和准备数据，以及构建 ``tf.data.Dataset``

      - 每一个数据集(dataset) 都实现了抽象基类 tfds.core.DatasetBuilder 来构建

   - 官方文档

      - https://github.com/tensorflow/datasets
      - https://www.tensorflow.org/datasets/overview
      - https://www.tensorflow.org/datasets/catalog/overview#all_datasets
      - https://www.tensorflow.org/datasets/api_docs/python/tfds
      - https://blog.tensorflow.org/2019/02/introducing-tensorflow-datasets.html?hl=zh-CN

2.3.1 查看可用的数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. code-block:: python

      import tensorflow as tf
      import tensorflow_datasets as tfds

      # 所有可用的数据集
      print(tfds.list_builders()) 

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

2.3.2 内置数据集分类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

2.3.3 构建并加载内置数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - ``tfds.load`` 是构建并加载 ``tf.data.Dataset`` 最简单的方式
   - ``tf.data.Dataset`` 是构建输入流水线的标准 TensorFlow 接口
   - 加载数据集时，默认使用规范的版本，但是可以指定要使用的数据集的主版本，并在结果中表明使用了哪个版本的数据集

示例1:

   .. code-block:: python

      mnist_train = tfds.load("mnist", split = "train", download = False, data_dir = "~/.tensorflow_datasets/")
      assert isinstance(mnist_train, tf.data.Dataset)
      print(mnist_train)

示例2：版本控制

   .. code-block:: python
   
      mnist = tfds.load("mnist:1.*.*")

2.3.4 内置数据集特征字典
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 所有 ``tensorflow_datasets, tfds`` 数据集都包含将特征名称映射到 Tensor 值的特征字典。典型的数据集将具有 2 个键:

      - ``"image"``
      - ``"label"``

示例:

   .. code-block:: python

      mnist_train = tfds.load("mnist", split = "train", download = False, data_dir = "~/.tensorflow_datasets/")
      for mnist_example in mnist_train.take(1):
         image, label = mnist_example["image"], mnist_example["label"]
         plt.imshow(
            image.numpy()[:, :, 0].astype(np.float32),
            cma = plt.get_cmap("gray")
         )
         print("Label: %d" % label.numpy())
         plt.show()

2.3.5 DatasetBuilder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - ``tensorflow_datasets.load`` 实际上是一个基于 ``DatasetBuilder`` 的简单方便的包装器

示例:

   .. code-block:: python

      mnist_builder = tfds.builder("mnist")
      mnsit_builder.download_and_prepare()
      mnist_train = mnist_builder.as_dataset(split = "train")
      mnist_train

2.3.6 内置数据集输入流水线
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 一旦有了 ``tf.data.Dataset`` 对象，就可以使用 ``tf.data`` 接口定义适合模型训练的输入流水线的其余部分.

示例:

   .. code-block:: python

      mnist_train = mnist_train.repeat().shuffle(1024).batch(32)

      # prefetch 将使输入流水线可以在模型训练时一步获取批处理
      mnist_train = mnist_train \
                     .repeat() \
                     .shuffle(1024) \
                     .batch(32) \
                     .prefetch(tf.data.experimental.AUTOTUNE)

2.3.7 内置数据集信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


2.3.9 内置数据集可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

示例:

   .. code-block:: python

      fig = tfds.show_examples(info, mnist_test)

3.TensorFlow Dataset 预处理数据
-------------------------------------

3.1 数据集预处理
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

4.3 文本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.4 CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.5 Numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.6 pandas.DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.7 Unicode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.8 TF.Text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4.9 TFRecord
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.9.1 TFRecord 数据文件介绍
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

4.9.2 TFRecord 文件保存
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



4.9.3 TFRecord 文件读取
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

4.10 tf.io 的其他格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



4.11 tf.TensorArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.11.1 tf.TensorArray 介绍
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

4.11.2 tf.TensorArray 介绍
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

