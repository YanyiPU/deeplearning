#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-16 00:07:29
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import xlrd


####################################################################
# Defing flags
####################################################################
# event file 路径设置
tf.app.flags.DEFINE_string(
	'log_dir', 
	os.path.dirname(os.path.abspath(__file__)) + '/logs', 
	'Directory where event logs are written to.'
)

tf.app.flags.DEFINE_integer(
	'num_epochs',
	50,
	'The number of epochs for training the model. Default=50'
)

FLAGS = tf.app.flags.FLAGS

if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
	raise ValueError("You must assign absolute path for --log_dir")


####################################################################
# Load dataset
####################################################################
# Dataaaa file from https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/2017/data
DATA_FILE = 'E:/DataScience/deeplearning/data/fire_theft.xls'
book = xlrd.open_workbook(DATA_FILE, encoding_override = 'utf-8')
sheet = book.sheet_by_index(0)

data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
num_samples = sheet.nrows - 1


####################################################################
# 
####################################################################
weights = tf.Variable(0.0, name = 'weights')
bias = tf.Variable(0.0, name = 'bias')


####################################################################
# 
####################################################################
def inputs():
	"""
	Defining the place_holders.
	:return:
		Returning the data and label lace holders.
	"""
	X = tf.placeholder(tf.float32, name = "X")
	Y = tf.placeholder(tf.float32, name = "Y")

	return X, Y


def inference():
	"""
	Forward passing the X.
	:param X: Input.
	:return: X*weights + bias
	"""

	return X * weights + bias


def loss(X, y):
	"""
	compute the loss by comparing the predicted value to the actual label.
	:param X: The input.
	:param Y: The label.
	:return: The loss over the samples.
	"""
	Y_pred = inference()
	return tf.squared_difference(Y, Y_pred)


def train(loss):
	learning_rate = 0.0001

	return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


####################################################################
# 
####################################################################
with tf.Session() as sess:
	# 初始化权重、偏置变量
	sess.run(tf.global_variables_initializer())

	# 输入Tensor
	X, Y = inputs()

	# 通过对平方损失函数应用随机梯度下降算法进行模型训练
	train_loss = loss(X, y)
	train_op = train(train_loss)

	for epoch_num in range(FLAGS.num_epochs):
		for x, y in data:
			train_op = train(train_loss)
			loss_value, _ = sess.run([train_loss, train_op], feed_dict = {X: x, Y: y})

		print("epoch %d, loss = %f" % (epoch_num + 1, loss_value))

		# 保存训练得到的模型权重参数和偏置
		weights_coeff, bias = sess.run([W, b])


####################################################################
# 
####################################################################
Input_values = data[:, 0]
Labels = data[:, 1]
Pred_values = data[:, 0] * weights_coeff + bias


plt.plot(Input_values, Labels, 'ro', label = "Original")
plt.plot(Input_values, Pred_values, label = "Predicted")
plt.legend()
plt.show()