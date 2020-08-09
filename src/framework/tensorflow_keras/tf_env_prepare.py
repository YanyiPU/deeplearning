#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-15 22:25:41
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

from __future__ import print_function
import tensorflow as tf
import os

# event file 路径设置
tf.app.flags.DEFINE_string(
	'log_dir', 
	os.path.dirname(os.path.abspath(__file__)) + '/logs', 
	'Directory where event logs are written to.'
)
FLAGS = tf.app.flags.FLAGS

if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
	raise ValueError("You must assign absolute path for --log_dir")
