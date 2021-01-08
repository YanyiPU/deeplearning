import tensorflow as tf
import time
from model.CNN import CNN
from utils.MNISTLoader import MNISTLoader


num_batches = 400
batch_size = 50
learning_rate = 0.001


@tf.function
def train_one_step(X, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred)
        loss = tf.reduce_mean(loss)
        # 注意这里使用了 TensorFlow 内置的 tf.print(), @tf.function 不支持 Python 内置的 print 方法
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))


if __name__ == "__main__":
    # data
    data_loader = MNISTLoader()
    # model
    model = CNN()
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    # model training
    start_time = time.time()
    
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y, optimizer)
    
    end_time = time.time()
    print(end_time - start_time)