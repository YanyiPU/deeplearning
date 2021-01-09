
Keras 数据集
============



1.[图像分类] CIFAR10
------------------------

-  数据

   -  Train set:

      -  50000 32x32 color images

         -  shape:

            -  ``image_data_format = "channels_first"``: (50000, 3, 32,
               32)

            -  ``image_data_format = "channels_last"``: (50000, 32, 32,
               3)

      -  10 categories

         -  shape: (50000,)

   -  Test set:

      -  10000 32x32 color images

         -  shape:

            -  ``image_data_format = "channels_first"``: (10000, 3, 32,
               32)

            -  ``image_data_format = "channels_last"``: (10000, 32, 32,
               3)

      -  10 categories

         -  shape: (10000,)
- 引用

   .. code:: python

      from keras.datasets import cifar10

      (x_train, y_train), (x_test, y_test) = cifar10.load_data()



2.[图像分类] CIFAR100
-------------------------

-  数据

   -  Train set:

      -  50000 32x32 color images

         -  shape:

            -  ``image_data_format = "channels_first"``: (50000, 3, 32,
               32)

            -  ``image_data_format = "channels_last"``: (50000, 32, 32,
               3)

      -  100 categories

         -  shape: (50000,)

   -  Test set:

      -  10000 32x32 color images

         -  shape:

            -  ``image_data_format = "channels_first"``: (10000, 3, 32,
               32)

            -  ``image_data_format = "channels_last"``: (10000, 32, 32,
               3)

      -  100 categories

         -  shape: (10000,)

- 引用

   .. code:: python

      from keras.datasets import cifar100

      (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode = "fine")



3.[文本分类] IMDB Movie review sentiment(IMDB电影评论情绪)
------------------------------------------------------------

-  数据

   -  Train set:

      -  25000 部电影评论

         -  每条评论被编码为一个单词索引序列，单词索引为单词在整个数据中的频率排序

      -  标签：情绪

         -  Positive

         -  Negitive

   -  Test set:

- 引用

   .. code:: python

      from keras.datasets import imdb

      (x_train, y_train), (x_test, y_test) = imdb.load_data(
         path = "imdb.npz",
         num_word = None,
         skip_top = 0,
         maxlen = None,
         seed = 113,
         start_char = 1,
         oov_char = 2,
         index_from = 3
      )



4.[文本分类] Reuters newswire topics(路透社新闻专题主题分类)
-------------------------------------------------------------

-  数据

   -  11228 新闻专线

      -  each wire is encoded as a sequence of word indexes

   -  46 主题

- 引用

   .. code:: python

      from keras.datasets import reuters

      (x_train, y_train), (x_test, y_test) = reuters.load_data(path = "reuters.npz",
         num_words = None, 
         skip_top = 0,
         maxlen = None,
         test_spilt = 0.2,
         seed = 113,
         start_char = 1,
         oov_char = 1,
         index_from = 3
      )

      # 用于编码序列的单词索引
      # word_index = {"word": index}
      word_index = reuters.get_word_index(path = "reuters_word_index.json")



5.[图像分类] MNIST
------------------

-  数据

   -  Train set:

      -  60000 28x28 grayscale images

      -  10 digits

   -  Test set:

      -  10000 28x28 grayscale images

      -  10 digits

- 引用

   .. code:: python

      from keras.datasets import mnist

      (x_train, y_train), (x_test, y_test) = mnist.load_data(path = "~/.keras/datasets/")



6.[图像分类] Fashion-MNIST
------------------------------

-  数据

   -  Train set:

      -  60000 28x28 grayscale images

      -  10 fashion categories

   -  Test set:

      -  10000 28x28 grayscale images

      -  10 fashion categories

-  类别标签

   +-------+-------------+
   | Label | Description |
   +=======+=============+
   | 0     | T-shirt/top |
   +-------+-------------+
   | 1     | Trouser     |
   +-------+-------------+
   | 2     | Pullover    |
   +-------+-------------+
   | 3     | Dress       |
   +-------+-------------+
   | 4     | Coat        |
   +-------+-------------+
   | 5     | Sandal      |
   +-------+-------------+
   | 6     | Shirt       |
   +-------+-------------+
   | 7     | Sneaker     |
   +-------+-------------+
   | 8     | Bag         |
   +-------+-------------+
   | 9     | Ankle boot  |
   +-------+-------------+

- 引用

   .. code:: python

      from keras.datasets import fashion_mnist

      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



7.[结构化数据回归] Boston housing price
------------------------------------------

-  数据

   -  特征个数: 13

   -  目标变量: median values of the houses at a location(in k$)

- 引用

   .. code:: python

      from keras.datasets import boston_housing

      (x_train, y_train), (x_test, y_test) = boston_housing.load_data(path = "~/.keras/datasets", seed, test_split)
