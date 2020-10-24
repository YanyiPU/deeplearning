
# Keras-Linear-Regression

# 导入模块


import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential, Model



# 创建数据

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]



# 建立模型

model = Sequential()
model.add(Dense(output_dim = 1, input_dim = 1))


# 编译模型

model.compile(loss = "mse", optimizer = "sgd")


# 训练模型

# training 
print("Training -------------")
for step in range(301):
   cost = model.train_on_batch(X_train, Y_train)
   if step % 100 == 0:
      print("train cost: ", cost)


# 验证模型

# test 
print("\nTesting -------------")
cost = model.evaluate(X_test, Y_test, batch_size = 40)
print("test cost: ", cost)
W, b = model.layers[0].get_weight()
print("Weights = ", W)
print("Biases = ", b)


# 模型预测及模型结果可视化
# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_test)
plt.show()
