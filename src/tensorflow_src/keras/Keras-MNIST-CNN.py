
# Keras-MNIST-CNN

# 1. Keras Sequential Model

"""
Trains a simple convnet on the MNIST dataset

Gets to 99.25% test accuracy after 12 epochs
16 seconds per epoch on a GRID K520 GPU
"""

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras import utils
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data preprocessing
X_train = X_train.reshape(-1, 1, 28, 28) / 255.
X_test = X_test.reshape(-1, 1, 28, 28) / 255.
y_train = utils.to_categorical(y_train, num_classes = 10)
y_test = utils.to_categorical(y_test, num_classes = 10)

model = Sequential()
model.add(Conv2D(
   batch_input_shape = (None, 1, 28, 28),
   filters = 32,
   kernel_size = 5,
   strides = 1,
   padding = "same",
   data_format = "channels_first",
))
model.add(Activation("relu"))
model.add(MaxPooling2D(
   pool_size = 2,
   strides = 2,
   padding = "same",
   data_format = "channels_first",
))
model.add(Conv2D(
   64, 
   kernel_size = 5, 
   strides = 1,
   padding = "same",
   data_format = "channels_first",
))
model.add(Activation("relu"))
model.add(MaxPooling2D(
   2, 
   2, 
   padding = "same", 
   data_format = "channels_first"
))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

adam = Adam(lr = 1e-4)
model.compile(optimizer = adam,
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])

print("Training ------------")
model.fit(X_train, y_train, epochs = 1, batch_size = 64)

print("\nTesting -------------")
loss, accuracy = model.evaluate(X_vaild, y_valid)

print("\nPredicting -------------")
y_pred = model.predict(X_test)

print("\ntest loss: ", loss)
print("\ntest accuracy: ", accuracy)


# 2. Keras 函数式 API Model

