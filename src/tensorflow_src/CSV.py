# import functools
# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds

# np.set_printoptions(precision = 3, suppress = True)

# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("test.csv", TEST_DATA_URL)


import pandas as pd

df = pd.read_csv("/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/tensorflow_src/2020-11-01 00_00_00 µ½ 2020-11-01 12_59_59ExportOrderDetailList202012021529.csv", header = 0, encoding = "gb2312")
print(df.head())