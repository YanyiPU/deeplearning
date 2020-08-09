#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


base_dir = "/Users/zfwang/project/deeplearning_project/deeplearning/tensorflow"
models_dir = os.path.join(base_dir, "models")
images_dir = os.path.join(base_dir, "images")



# an encoder model that turns image inputs into 16-dimensional vectors
encoder_input = keras.Input(shape = (28, 28, 1), name = "img")
x = layers.Conv2D(16, 3, activation = "relu")(encoder_input)
x = layers.Conv2D(32, 3, activation = "relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation = "relu")(x)
x = layers.Conv2D(16, 3, activation = "relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name = "encoder")
encoder.summary()

# an end-to-end autoencoder model for training
x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
x = layers.Conv2DTranspose(32, 3, activation = "relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation = "relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name = "autoencoder")
autoencoder.summary()
images_path_1 = os.path.join(images_dir, "autoencoder_1.png")
keras.utils.plot_model(autoencoder, images_path_1, show_shapes = True)




# ----------------------------------------------------------------------
encoder_input = keras.Input(shape = (28, 28, 1), name = "original_img")
x = layers.Conv2D(16, 3, activation = "relu")(encoder_input)
x = layers.Conv2D(32, 3, activation = "relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation = "relu")(x)
x = layers.Conv2D(16, 3, activation = "relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name = "encoder")
encoder.summary()

decoder_input = keras.Input(shape = (16,), name = "encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
x = layers.Conv2DTranspose(32, 3, activation = "relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation = "relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation = "relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name = "decoder")
decoder.summary()

autoencoder_input = keras.Input(shape = (28, 28, 1), name = "img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name = "autoencoder")
autoencoder.summary()
images_path_2 = os.path.join(images_dir, "autoencoder_2.png")
keras.utils.plot_model(autoencoder, images_path_2, show_shapes = True)

dot_model = keras.utils.model_to_dot(autoencoder, show_shapes = True, subgraph = True)
print(dot_model)