#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


base_dir = "/Users/zfwang/project/deeplearning_project/mnist"
models_dir = os.path.join(base_dir, "models")
images_dir = os.path.join(base_dir, "images")

# data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

# model
inputs = keras.Input(shape = (784,))
dense = layers.Dense(64, activation = "relu")(inputs)
x = layers.Dense(74, activation = "relu")(dense)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs = inputs, outputs = outputs, name = "mnist_model")
model.summary()
keras.utils.plot_model(model, os.path.join(images_dir, "my_first_model.png"))
keras.utils.plot_model(model, os.path.join(images_dir, "my_first_model_with_shape_info.png"), show_shapes = True)

# model compile
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.RMSprop(),
    metrics = ["accuracy"],
)

# model training
history = model.fit(x_train, y_train, batch_size = 64, epochs = 2, validation_split = 0.2)

# model evaluate
test_scores = model.evaluate(x_test, y_test, verbose = 2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# model save and serialize
model_path = os.path.join(models_dir, "mnist_functional_api_dnn.h5")
model.save(model_path)
del model

# recreate the exact same model purely from the file
model = keras.models.load_model(model_path)
model.summary()
