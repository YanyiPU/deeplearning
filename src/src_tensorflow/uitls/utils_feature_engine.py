# -*- coding: utf-8 -*-
import tensorflow as tf


def make_input_function(data_df, label_df, num_epochs = 10, batch_size = 256, training = True):
    """
    An input function for training or evaluating
    """
    def input_function():
        """
        将输入转换为 tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        # 如果在训练模式下混淆并重复数据
        if training:
            dataset = dataset.shuffle(1000).repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        return dataset

    return input_function


def indicator_cat_column(feature_name, vocab):
    """
    将类别型变量转换为整数标签
    """
    return tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)


def one_hot_cat_column(feature_name, vocab):
    """
    类别型变量 One-Hot 编码
    """
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)
    )


def make_numeric_column(feature_name):
    return tf.feature_column.numeric_column(feature_name, dtype = tf.float32)