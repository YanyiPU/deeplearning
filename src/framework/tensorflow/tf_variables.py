from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.python.framework import ops

# event file 路径设置
tf.app.flags.DEFINE_string(
    'log_dir', 
    os.path.dirname(os.path.abspath(__file__)) + '/logs', 
    'Directory where event logs are written to.'
)
FLAGS = tf.app.flags.FLAGS

if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError("You must assign absolute path for --log_dir")


# ======================================================================
# TensorFlow Variables
# ======================================================================
# ==============
# Create variables
# ==============
weights = tf.Variable(tf.random_normal([2, 3], stddev = 0.1), name = 'weights')

biases = tf.Variable(tf.zeros([3]), name = 'biases')

custom_variable = tf.Variable(tf.zeros([3]), name = 'custom')

variable_list_custom = [weights, custom_variable]

all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

WeightsNew = tf.Variable(weights.initialized_value(), name = 'WeightsNew')


# ==============
# Initialization
# ==============
# Initializing specific variables(custom initialization)
init_custom_op = tf.variables_initializer(var_list = variable_list_custom)

# Global variable initialization
init_all_op_method_1 = tf.global_variables_initializer()
init_all_op_method_2 = tf.variables_initializer(var_list = all_variables_list)

# Initialization of a variables using other existing variables
init_WeightsNew_op = tf.variables_initializer(var_list = [WeightsNew])


with tf.Session() as sess:
    sess.run(init_custom_op)
    sess.run(init_all_op_method_1)
    sess.run(init_WeightsNew_op)
