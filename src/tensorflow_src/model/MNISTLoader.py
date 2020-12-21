import numpy as np
import tensorflow as tf


"""
使用 tf.keras.datasets 获得数据集并预处理
实现一个简单的 MNISTLoader 类来读取 MNIST 数据集数据
"""


class MNISTLoader():
    def __init__(self):
        """
        Returns:
            self.train_data
            self.train_label
            self.test_data
            self.test_label
            self.num_train_data
            self.num_test_data
        """
        # minst data
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        
        # MNIST 中的图像默认为 uint8(0~255的数字)。以下代码将其归一化为 0~1 的浮点数，并在最后增加一维作为颜色通道
        # train data
        # [60000, 28, 28, 1]
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis = 1)
        self.train_label = self.train_label.astype(np.int32) # [60000]
        
        # test data
        # [10000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis = 1)
        self.test_label = self.test_label.astype(np.int32) # [10000]
        
        # trian_data, test_data number
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
    

    def get_batch(self, batch_size):
        """
        从数据中随机取出 batch_size 个元素并返回

        Args:
            batch_size ([type]): [description]

        Returns:
            [type]: [description]
        """
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)

        return self.train_data[index, :], self.train_label[index]



if __name__ == "__main__":
    data_loader = MNISTLoader()
    train_num = data_loader.num_train_data
    print(train_num)