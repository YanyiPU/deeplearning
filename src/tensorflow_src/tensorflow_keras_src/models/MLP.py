import numpy as np
import tensorflow as tf
# import MNISTLoader


print(tf.__version__)


"""
- 目标：使用多层感知机模型完成 MNIST 手写数字图片数据集 LeCun1998 的分类任务
- 数据： MNIST 
- 流程
    (1)使用 tf.keras.datasets 获得数据集并预处理
    (2)使用 tf.keras.Model 和 tf.keras.layers 构建模型
    (3)构建模型训练过程，使用 tf.keras.losses 计算损失函数，并使用 tf.keras.optimizer 优化模型
    (4)构建模型评估过程，使用 tf.keras.metrics 计算评估指标
"""


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten() # Flatten 层将除第一维(batch_size)以外的维度”展平“
        self.dense1 = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 10)
    
    @tf.function
    def call(self, input):        # [batch_size, 28, 28, 1]
        x = self.flatten(input)   # [batch_size, 28 * 28 * 1 = 784]
        x = self.dense1(x)        # [batch_size, 100]
        x = self.dense2(x)        # [batch_size, 10]
        output = tf.nn.softmax(x) # []
        return output


if __name__ == "__main__":
    num_epochs = 5
    batch_size = 50
    learning_rate = 1e-4
    
    # data
    data_loader = MNISTLoader.MNISTLoader()

    # model
    model = MLP()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate =  learning_rate)

    # batches
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)

    # model training
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred))
            # loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true = tf.one_hot(y, depth = tf.shape(y_pred)[-1]), y_pred = y_pred))
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

    # metric
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index:end_index])
        sparse_categorical_accuracy.update_state(
            y_true = data_loader.test_label[start_index:end_index], 
            y_pred = y_pred
        )
    print("test accuracy: %f" % sparse_categorical_accuracy.result())
