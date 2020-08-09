#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3
from tensorflow.keras import backend as K
# 禁用所有与模型训练有关的操作
K.set_learning_phase(0)


base_dir = os.path.dirname(__file__)
images_dir = os.path.join(base_dir, "images")
models_dir = os.path.join(base_dir, "models")
print(base_dir)




# model
model = inception_v3.InceptionV3(
    weights = "imagenet", # 使用预训练的 ImageNet 权重加载模型 
    include_top = False  # 不包括全连接层
)
keras.utils.plot_model(model, os.path.join(images_dir, "inception_v3.png"), show_shapes = True)
