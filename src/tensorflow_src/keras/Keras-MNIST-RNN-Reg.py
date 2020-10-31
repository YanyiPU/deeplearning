
# Keras-MNIST-RNN-Reg

# 1. Keras Sequential Model

# 导入模块

# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam


# 数据预处理


TIME_STEPS = 20
INPUT_SIZE = 1
BATCH_SIZE = 50
BATCH_START = 0
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data preprocessing
X_train = X_train.reshape(-1, 1, 28, 28) / 255.
X_test = X_test.reshape(-1, 1, 28, 28) / 255.
y_train = utils.to_categorical(y_train, num_classes = 10)
y_test = utils.to_categorical(y_test, num_classes = 10)


# 建立模型


# RNN model
model = Sequential()
# LSTM RNN
model.add(LSTM(
   batch_input_shape = (BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
   output_dim = CELL_SIZE,
   return_sequences = True,
   stateful = True
))
# output layer
model.add(TimeDistributed(
   Dense(OUTPUT_SIZE)
))

# 模型编译

adam = Adam(LR)
model.compile(optimizer = adam, loss = "mse")


# 模型训练

def get_batch():
   global BATCH_START, TIME_STEPS
   xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
   seq = np.sin(xs)
   res = np.cos(xs)
   BATCH_START += TIME_STEPS
   return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

print("Training ------------")
for step in range(501):
   X_batch, Y_batch, xs = get_batch()
   cost = model.train_on_batch(X_batch, Y_batch)
   pred = model.predict(X_batch, Y_batch)
   plt.plot(xs[0, :], Y_batch[0].flatten(), "r", xs[0, :], pred.flatten()[:TIME_STEPS], "b--")
   plt.ylim((-1.2, 1.2))
   plt.draw()
   plt.pause(0.1)
   if step % 10 == 0:
      print("train cost: ", cost)


# 模型评估

import keras


# 模型预测

import keras


# 模型结果输出

import keras


# 2. Keras 函数式 API Model

