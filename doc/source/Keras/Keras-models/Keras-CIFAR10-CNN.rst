.. _header-n0:

Keras-CIFAR10-CNN
=================

.. code:: python

   # Train a simple deep CNN on the CIFAR10 small images dataset.

   """
   It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
   (it's still underfitting at that point, though)
   """

   from __future__ import print_function

   import keras
   from keras import utils
   from keras.datasets import cifar10
   from keras.preprocessing.image import ImageDataGenerator
   from keras.models import Sequential
   from keras.layers import Dense, Dropout, Activation, Flatten
   from keras.layers import Conv2D, MaxPooling2D
   from keras import optimizers, losses, metrics
   import os

   batch_size = 32
   num_classes = 10
   epochs = 100
   data_augmentation = True
   num_predictions = 20
   save_dir = os.path.join(os.getcwd(), "save_models")
   model_name = "keras_cifar10_trained_model.h5"


   # data
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   x_train = x_train.astype("float32")
   x_test = x_test.astype("float32")
   x_train /= 255
   x_test /= 255
   y_train = utils.to_categorical(y_train, num_classes)
   y_test = utils.to_categorical(y_test, num_classes)
   print("x_train shape:", x_train.shape)
   print(x_train.shape[0], "train samples")
   print(x_test.shape[0], "test samples")


   # model
   model = Sequential()
   model.add(Conv2D(32, (3, 3), padding = "same",input_shape = x_train.shape[1:]))
   model.add(Activation("relu"))
   model.add(Conv2D(32, (3, 3)))
   model.add(Activation("relu"))
   model.add(MaxPooling2D(pool_size = (2, 2)))
   model.add(Dropout(0.25))
   model.add(Flatten())
   model.add(Dense(512))
   model.add(Activation = "relu")
   model.add(Dropout(0.5))
   model.add(Dense(num_classes))
   model.add(Activation("softmax"))


   # model compile
   rmsprop = optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
   model.compile(loss = "categorical_crossentropy",
                 optimizer = rmsprop,
                 metrics = ["accuracy"])


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
       model.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size = batch_size,
                                       epochs = epochs,
                                       validation_data = (x_test, y_test),
                                       workers = 4))

   # save model and weights
   if not os.path.isdir(save_dir):
       os.makedirs(save_dir)
   model_path = os.path.join(save_dir, model_name)
   model.save(model_path)
   print("Saved trained model at %s" % model_path)


   # Score trained model.
   scores = model.evaluate(x_test, y_test, verbose = 1)
   print("Test loss: ", scores[0])
   print("Test accuracy: ", scores[1])
