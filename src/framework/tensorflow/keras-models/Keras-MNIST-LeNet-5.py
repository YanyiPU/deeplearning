
# Keras-MNIST-LeNet-5

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import keras 
from keras.layers import Conv2D, Dense, Dropout, Activation, MaxPooling2D, Flatten
from keras.model import Sequential


# LeNet-5 model
def lenet5(x_train, y_train):
model = Sequential()

model.add(Conv2D(6, (5, 5), 
                strides = 1, 
                padding = "valid", 
                input_shape = (32, 32, 1),
                activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(16, (5, 5),
                    strides = 1,
                    activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(1203, (5, 5), 
                    strides = 1, 
                    activation = "relu"))
model.add(Flatten())
model.add(Dense(84))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation("softmax"))

mode.compile(loss = "categorical_crossentropy",
                optimizer = "adam",
                metrics = ["accuracy"])

model.fit(x_train, y_train)

return model

def eval_score(model, x_test, y_test):
score = model.evaluate(x_test, y_test, batch_size = 128)
return score

def main():
# data
mnsit = input_data.read_data_sets("MNIST_data", one_hot = True)
session = tf.InteractiveSession()
x_train = mnist.train.images
x_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]]).eval()
y_train = mnist.train.labels
x_test = mnist.test.images
x_test = tf.reshape(x_test, [-1, 28, 28, 1])
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]]).eval()
y_test = mnist.test.labels 

# model training
model = lenet5(x_train, y_train)
print(model.summary())

# model evaluation
score = eval_score(model, x_test, y_test)
print(score)

if __init__ == "__main__":
main()
