import os
import tensorflow as tf
from model.MLP import MLP
from utils.MNISTLoader import MNISTLoader


num_batches = 1000
batch_size = 50
learning_rate = 1e-3

# root path
project_root_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/"
root_path = os.path.join(project_root_path, "src/tensorflow_src")
log_dir = os.path.join(root_path, "tensorboard")


# data
data_loader = MNISTLoader()
# model
model = MLP()
# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
# Tensorboard
summary_writer = tf.summary.create_file_writer(log_dir)
tf.summary.trace_on(profiler = True) # 开启 Trace

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred))
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step = batch_index)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

with summary_writer.as_default():
    # 保存 Trace 信息到文件(可选)
    tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = log_dir)
