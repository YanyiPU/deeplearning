#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


# -------------------------
# build model
# -------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1))) # (26, 26, 32)
model.add(layers.MaxPooling2D((2, 2)))                                               # (13, 13, 32)
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))                            # (11, 11, 64)
model.add(layers.MaxPooling2D((2, 2)))                                               # (5, 5, 64)
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))                            # (3, 3, 64)
model.add(layers.Flatten())                                                          # (576,)
model.add(layers.Dense(64, activation = "relu"))                                     # (64,)
model.add(layers.Dense(10, activation = "softmax"))                                  # (10,)

# -------------------------
# model compile
# -------------------------
model.compile(optimizer = "rmsprop",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# -------------------------
# data preprocessing 
# -------------------------
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype("float32") / 255
test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# -------------------------
# train model
# -------------------------
model.fit(train_data, train_labels, epochs = 5, batch_size = 64)

# -------------------------
# validation model
# -------------------------
test_loss, test_acc = model.evaluate(test_data, test_labels)


if __name__ == "__main__":
    model.summary()
    print("test_acc:", test_acc)