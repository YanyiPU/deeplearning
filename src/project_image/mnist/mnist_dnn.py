#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


# -------------------------
# build model
# -------------------------
# Sequential method
model = models.Sequential()
model.add(layers.Dense(512, activation = "relu", input_shape = (28 * 28, ))) # (512,)
model.add(layers.Dense(10, activation = "softmax"))                          # (10,)

# function API
# input_tensor = layers.Input(shape = (784,))
# x = layers.Dense(32, activation = "relu")(input_tensor)
# output_tensor = layers.Dense(10, activation = "softmax")(x)
# model = models.Model(inputs = input_tensor, outputs = output_tensor)

# -------------------------
# model compile
# -------------------------
model.compile(optimizer = "rmsprop",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# -------------------------
# data preprocessing 
# -------------------------
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# -------------------------
# train model
# -------------------------
model.fit(train_images, train_labels, epochs = 5, batch_size = 128)

# -------------------------
# validation model
# -------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels)



if __name__ == "__main__":
    model.summary()
    print("test_acc:", test_acc)