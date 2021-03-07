import numpy as np
import tensorflow as tf


def vectorize_sequences(sequences, dimension = 10000):
    """
    将数据向量化
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequnece in enumerate(sequences):
        results[i, sequences] = 1.
        
        return results


def to_one_hot(labels, dimension = 46):
    """
    # 将标签向量化
    """
    results = np.zeros(len(labels), dimension)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    
    return results


def rot90(image, label):
    """
    将图片旋转 90 度
    """
    image = tf.image.rot90(image)
    return image, label





if __name__ == "__main__":
    train_data = "wangzhefeng wang zhe feng"
    x_train = vectorize_sequences(train_data)
    print(x_train)
    # x_test = vectorize_sequences(test_data)

    # one_hot_train_labels = to_one_hot(train_label)
    # one_hot_test_labels = to_one_hot(test_label)