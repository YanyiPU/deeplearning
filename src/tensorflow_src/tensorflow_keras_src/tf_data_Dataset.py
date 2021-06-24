import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# --------------------------------------------------------
# Transform numpy or tensorflow array to tf.data.Dataset
# --------------------------------------------------------
# numpy variable
X_numpy = np.array([2013, 2014, 2015, 2016, 2017])
Y_numpy = np.array([12000, 14000, 15000, 16500, 17500])

# tensorflow variable
X_tf = tf.constant([2013, 2014, 2015, 2016, 2017])
Y_tf = tf.constant([12000, 14000, 15000, 16500, 17500])

dataset = tf.data.Dataset.from_tensor_slices((X_numpy, Y_numpy))
for x, y in dataset:
    print(x.numpy(), y.numpy())



# --------------------------------------------------------
# Load MNIST data to tf.data.Dataset
# --------------------------------------------------------
(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis = -1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()


# --------------------------------------------------------
# tf.data.Dataset 数据预处理
# --------------------------------------------------------
# Dataset.map(f)
# Dataset.shuffle(buffer_size)
# Dataset.batch(batch_size)
# Dataset.repeat()
# Dataset.reduce()
# Dataset.take()
# Dataset.prefetch()


def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label

mnist_dataset = mnist_dataset.map(rot90)

for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()


