

Keras 数据预处理
================



内容
----

-  Sequence Preprocessing

   -  TimeseriesGenerator

   -  pad_sequences

   -  skipgrams

   -  make\ *sampling*\ table

-  Text Preprocessing

   -  Tokenizer

   -  hashing_trick

      -  将文本转换为固定大小的散列空间中的索引序列

   -  one_hot

      -  One-hot将文本编码为大小为n的单词索引列表

   -  text\ *to*\ word_sequence

      -  将文本转换为单词（或标记）序列

-  Image Preprocessing

   -  ``class`` ImageDataGenerator

      -  ``method``

         -  .apply_transform()

         -  .fit ()

         -  .flow()

            -  采用数据和标签数组，生成批量增强数据

         -  .flow\ *from*\ dataframe()

            -  获取数据帧和目录路径，并生成批量的扩充/规范化数据

         -  .flow\ *from*\ directory()

            -  获取目录的路径并生成批量的增强数据

         -  .get\ *random*\ transform()

            -  为转换生成随机参数

         -  .random_transform()

            -  随机转换

         -  .standardize()

            -  标准化




1.Sequence Preprocessing 
-------------------------



2.Text Preprocessing
--------------------

.. code:: python

   from keras.preprocessing.text import Tokenizer
   from keras.preprocessing.text import hashing_trick
   from keras.preprocessing.text import one_hot
   from keras.preprocessing.text import text_to_word_sequence



3.Image Preprocessing
---------------------

**keras.preprocessing.imgae.ImageDataGenerator
通过实时数据增强生成批量张量图像数据：**

.. code:: python

   keras.preprocessing.image.ImageDataGenerator(featurewise_center = False, # 将数据的特征均值设定为0
       samplewise_center = False,  # 将数据的样本均值设定为0
       featurewise_std_normalization = False, # 是否将特征除以特征的标准差进行归一化
       samplewise_std_normalization = False,  # 是否将样本除以样本的标准差进行归一化
       zca_whitening = False, # 是否进行 ZCA 白化
       zca_epsilon = 1e-06,   # 进行 ZCA 白化的epsilon参数
       rotation_range = 0,      # 随机旋转的角度范围
       width_shift_range = 0.0, # 宽度调整的范围
       height_shift_range = 0.0,# 高度调整的范围
       brightness_range = None, # 亮度范围 
       shear_range = 0.0,         # 剪切范围
       zoom_range = 0.0,          # 缩放范围
       channel_shift_range = 0.0, # 通道调整范围
       fill_mode = 'nearest',     # 填充边界之外点的方式:
       cval=0.0, 
       horizontal_flip=False,  # 水平翻转
       vertical_flip=False,    # 垂直翻转
       rescale=None,           # 
       preprocessing_function=None, 
       data_format=None, 
       validation_split=0.0,
       dtype=None)

**用法：**

.. code:: python

   from keras.datasets import cifar10
   from keras import utils
   from keras.preprocessing.image import ImageDataGenerator

   # model training parameters
   num_classes = 10
   data_augmentation = True
   batch_size = 32
   epochs = 20

   # data
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   x_train = x_train.astype("float32")
   x_test = x_test.astype("float32")
   x_train /= 255
   x_test /= 255
   y_train = utils.to_categorical(y_train, num_classes = num_classes)
   y_test = utils.to_categorical(y_test, num_classes = num_classes)

   # model training
   if not data_augmentation:
       print("Not using data augmentation.")
       model.fit(x_train, y_train,
                 batch_size = batch_size,
                 epochs = epochs,
                 validation_data = (x_test, y_test),
                 shuffle = True)
   else:
       print("Using real-time data augmentation.")
       # This will do preprocessing and realtime data augmentation:
       datagen = ImageDataGenerator(
           featurewise_center = False,
           samplewise_center = False,
           featurewise_std_normalization = False,
           samplewise_std_normalization = False,
           zca_whitening = False,
           zca_epsilon = 1e-6,
           rotation_range = 0,
           width_shift_range = 0.1,
           height_shift_range = 0.1,
           shear_range = 0.,
           zoom_range = 0.,
           channel_shift_range = 0,
           fill_mode = "nearest",
           cval = 0.,
           horizontal_flip = True,
           vertical_flip = False,
           rescale = None,
           preprocessing_function = None,
           data_format = None,
           validation_split = 0.0
       )
       datagen.fit(x_train)
       model.fit_generator(datagen.flow(x_train,
                                        y_train,
                                        batch_size = batch_size,
                                        epochs = epochs,
                                        validation_data = (x_test, y_test),
                                        workers = 4))

.. code:: python

   from keras.datasets import cifar10
   from keras import utils


   # data
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   x_train = x_train.astype("float32")
   x_test = x_test.astype("float32")
   x_train /= 255
   x_test /= 255
   y_train = utils.to_categorical(y_train, num_classes = num_classes)
   y_test = utils.to_categorical(y_test, num_classes = num_classes)


   # model training parameters
   batch_size = 32
   epochs = 20
   num_classes = 10
   data_augmentation = True

   # model training
   datagen = ImageDataGenerator(featurewise_center = True,
                                featurewise_std_normalization = True,
                                rotation_range = 20,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                horizontal_flip = True)

   for e in range(epochs):
       print("Epoch", e)
       batches = 0
       for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size = batch_size):
           model.fit(x_batchd, y_batch)
           batches += 1
           if batches >= len(x_train) / 32:
               break

.. code:: python

   train_datagen = ImageDataGenerator(rescale = 1. / 255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)
   test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

   train_generator = train_datagen \
       .flow_from_directory("data/train",
                            target_size = (150, 150),
                            batch_size = 32,
                            class_mode = "binary")
   validation_generator = test_datagen \
       .flow_from_directory("data/validation",
                            target_size = (150, 150),
                            batch_size = 32,
                            class_mode = "binary")

   model.fit_generator(train_generator,
                       steps_per_epoch = 2000,
                       epochs = 50,
                       validation_data = validation_generator,
                       validation_steps = 800)

.. code:: python

   # we create two instances with the same arguments
   data_gen_args = dict(featurewise_center=True,
                        featurewise_std_normalization=True,
                        rotation_range=90,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)
   image_datagen = ImageDataGenerator(**data_gen_args)
   mask_datagen = ImageDataGenerator(**data_gen_args)

   # Provide the same seed and keyword arguments to the fit and flow methods
   seed = 1
   image_datagen.fit(images, augment=True, seed=seed)
   mask_datagen.fit(masks, augment=True, seed=seed)

   image_generator = image_datagen.flow_from_directory(
       'data/images',
       class_mode=None,
       seed=seed)

   mask_generator = mask_datagen.flow_from_directory(
       'data/masks',
       class_mode=None,
       seed=seed)

   # combine generators into one which yields image and masks
   train_generator = zip(image_generator, mask_generator)

   model.fit_generator(
       train_generator,
       steps_per_epoch=2000,
       epochs=50)



