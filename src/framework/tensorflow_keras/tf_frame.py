#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings



#===========================================================
#               TensorFlow计算模型——计算图(Graph)
#===========================================================
# example 1
a = tf.constant([1.0, 2.0], name = "a", dtype = tf.float32)
b = tf.constant([2.0, 3.0], name = "b", dtype = tf.float32)
result1 = a + b
sess = tf.Session()
print(sess.run(result1))

print(a.graph is tf.get_default_graph())




# example 2 
"""
tf.Graph():
	生成新的计算图
"""
g1 = tf.Graph()
with g1.as_default():
	v = tf.get_variable("v", shape = [1], initializer = tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
	v = tf.get_variable("v", shape = [1], initializer = tf.ones_initializer)

with tf.Session(graph = g1) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("", reuse = True):
		print(sess.run(tf.get_variable("v")))

with tf.Session(graph = g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("", reuse = True):
		print(sess.run(tf.get_variable("v")))



# example 3
"""
tf.Graph.device():
	指定计算运行设备
"""
g = tf.Graph()

with g.device('/gpu:0'):
	result = a + b


"""
通过集合(collection)管理不同类别的资源

tf.add_to_collection()
tf.get_collection()

Tensorflow自动管理了一些常用的集合

tf.GraphKeys.VARIABLES
tf.GraphKeys.TRAINABLE_VARIABLES
tf.GraphKeys.SUMMARIES
tf.GraphKeys.QUEUE_RUNNERS
tf.GraphKeys.MOVING_AVERAGE_VARIABLES

"""





#===========================================================
#              TensorFlow数据模型——张量(Tensor)
#===========================================================
# example 1
a = tf.constant([1.0, 2.0], name = "a", dtype = tf.float32)
b = tf.constant([2.0, 3.0], name = "b", dtype = tf.float32)
result2 = tf.add(a, b, name = "add")
print(result2)
"""
输出：
Tensor("add_1:0", shape=(2,), dtype=float32)
"""



"""
TensorFlow 支持14中数据类型
tf.float32
tf.float64
tf.int8
tf.int16
tf.int32
tf.int64
tf.uint8
tf.bool
tf.complex64
tf.complex128
"""


#===========================================================
#              TensorFlow运行模型——回话(Session)
#===========================================================
# 一般模式
sess = tf.Session()
sess.run()
sess.close()


# Python上下文管理器自动管理会话
with tf.Session() as sess:
	sess.run()

# 
sess = tf.Session()
with sess.as_default():
	print(result.eval)

#
sess = tf.Session()
print(sess.run(result))
print(result.eval(session = sess))

#
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
