
# Keras-MNIST-Classification

# 导入模块

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras import utils
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from keras.optimizers import RMSprop


# 数据预处理

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data preprocessing
# normalize
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# one-hot encoding
y_train = utils.to_categorical(y_train, num_classes = 10)
y_test =  utils.to_categorical(y_test, num_classes = 10)

print(X_train[1].shape)
print(y_train[:3])


# 建立模型


model = Sequential([
   Dense(output_dim = 32, input_dim = 784),
   Activation("relu"),
   Dense(output_dim = 10)
   Activation("softmax")
])

# 编译模型

rmsprop = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer = rmsprop,
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])


# 训练模型


print("Training ---------------")
model.fit(X_train, y_train, epoch = 2, batch_size = 32)


# 评估模型

print("\nTesting ---------------")
loss, accuracy = model.evaluate(X_test, y_test)
print("test loss: ", loss)
print("test accuracy: ", accuracy)


# 模型预测


Y_pred = model.predict(X_test)
