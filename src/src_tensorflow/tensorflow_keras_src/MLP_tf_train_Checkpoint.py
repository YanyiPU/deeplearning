import os
import numpy as np
import tensorflow as tf
import argparse
from model.MLP import MLP
from utils.MNISTLoader import MNISTLoader

# root path
project_root_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/"
root_path = os.path.join(project_root_path, "src/tensorflow_src")


# 模型超参数
parser = argparse.ArgumentParser(description = "Process some integers.")
parser.add_argument("--mode", default = "train", help = "train or test")
parser.add_argument("--num_epochs", default = 1)
parser.add_argument("--batch_size", default = 50)
parser.add_argument("--learning_rate", default = 0.001)
args = parser.parse_args()

# data
data_loader = MNISTLoader()


def train():
    # Model build
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
    # Checkpoint
    checkpoint = tf.train.Checkpoint(myAwesomeModel = model)
    manager = tf.train.CheckpointManager(checkpoint, directory = os.path.join(root_path, "save"), checkpoint_name = "model.ckpt", max_to_keep = 10)
    # Model training
    for batch_index in range(1, num_batches + 1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
        if batch_index % 100 == 0:
            # version 1
            # path = checkpoint.save(os.path.join(root_path, "save/model.ckpt"))
            # version 2
            path = manager.save(checkpoint_number = batch_index)
            print("model saved to %s" % path)


def test():
    model_to_be_restored = MLP()
    # Checkpoint
    checkpoint = tf.train.Checkpoint(myAwesomeModel = model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint(os.path.join(root_path, "save")))
    y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis = -1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))


if __name__ == "__main__":
    # model
    if args.mode == "train":
        train()
    if args.mode == "test":
        test()
