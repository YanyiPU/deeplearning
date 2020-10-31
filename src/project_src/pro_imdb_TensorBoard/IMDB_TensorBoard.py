#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "wangzhefeng"


"""
 任务：IMDB 情感分析
 数据：IMDB(keras.datasets.imdb)
 模型：CNN
"""



import keras
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence


# 作为特征的单词个数
max_features = 2000
# 截断文本
max_len = 500

# data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
x_train = sequence.pad_sequences(x_train, maxlen = max_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_len)

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length = max_len, name = 'embed'))
model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))


model.summary()


model.compile(
    optimizer = "rmsprop",
    loss = 'binary_crossentropy',
    metrics = ['acc']
)

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'my_log_dir',
        histogram_freq = 1,
        embeddings_freq = 1
    )
]

history = model.fit(x_train, y_train,
                    epochs = 20,
                    batch_size = 128,
                    validation_split = 0.2,
                    callbacks = callbacks)



loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)