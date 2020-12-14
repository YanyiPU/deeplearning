# -*- coding: utf-8 -*-
# ----------------------
# 计算神经元输出--前向传播
# ----------------------
import numpy as np

example_input = [1, .2, .1, .05, .2]
example_weights = [.2, .12, .4, .6, .90]

input_vector = np.array(example_input)
weights = np.array(example_weights)
bias_weight = .2

activation_level = np.dot(input_vector, weights) + (bias_weight * 1)
print(activation_level)

# ----------------------
# 使用激活函数阈值
# ----------------------
threshold = 0.5
if activation_level >= threshold:
    perceptron_output = 1
else:
    perceptron_output = 0

print(perceptron_output)



# ----------------------
# 5.1.7
# ----------------------


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Out examples for an exclusive OR.
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
model = Sequential()
num_neurons = 10
model.add(Dense(num_neurons, input_dim = 2))
model.add(Activation("tanh"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()



# 
# 5.1 Neural networks, the ingredient list
# 5.1.1 Perceptron
# 5.1.2 A numerical perceptron
# 5.1.3 Detour through bias
# A python neuron
    # Class is in session
    # Logic is a fun thing to learn
    # Next step
    # Emergence from the second AI winter
    # Backpropagation
    # differentiate all the things

# 5.1.4 Let's go skilling-the error surface
# 5.1.5 Off the chair lift, onto the slope
# 5.1.6 Let's shake things up a bit
# 5.1.7 Keras: Neural networks in Python
# 5.1.8 Onward and deepward
# 5.1.9 Normalization: input with style

