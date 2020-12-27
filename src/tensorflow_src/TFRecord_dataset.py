import tensorflow as tf
import os


# root
root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
# project
project_path = os.path.join(root_dir, "deeplearning/src/tensorflow_src")
# model save
models_path = os.path.join(project_path, "save")
# data
cats_and_dogs_dir = os.path.join(root_dir, "datasets/cats_vs_dogs")
data_dir = os.path.join(root_dir, "datasets/cats_vs_dogs/cats_and_dogs_small")
# train data
train_dir = os.path.join(data_dir, "train")
train_cats_dir = os.path.join(train_dir, "cat")
train_dogs_dir = os.path.join(train_dir, "dog")
# tfrecord
tfrecord_file = os.path.join(cats_and_dogs_dir, "tfrecord")


# 训练数据
train_cat_filenames = [os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [os.path.join(train_dogs_dir, filename) for filenmae in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

# 迭代读取每张图片，建立 tf.train.Feature 字典和 tf.train.Example 对象，序列化并写入 TFRecord
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels)
    # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
    image = open(filename, "rb").read()
    feature = {
        
    }