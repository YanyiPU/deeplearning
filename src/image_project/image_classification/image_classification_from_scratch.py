"""
Author: fchollet
Date created: 2020/04/27
Last modified: 2020/04/28
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
"""


# 1.Setup
import config
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 2.Load the data: the Cats vs Dogs dataset
# (1)Raw data download
"""
# !curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
# !unzip -q kagglecatsanddogs_3367a.zip
# !ls
# !ls PetImages
"""

# (2)去除损坏的图像
def filter_out_corrupted_images():
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(config.data_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
    print(f"Deleted {num_skipped} images.")

# (3)创建 tf.Dataset
def get_dataset():
    image_size = (180, 180)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.data_dir,
        validation_split = 0.2,
        subset = "training",
        seed = 1337,
        image_size = image_size,
        batch_size = batch_size,
    )
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.data_dir,
        validation_split = 0.2,
        subset = "validation",
        seed = 1337,
        image_size = image_size,
        batch_size = batch_size,
    )
    return train_ds, validation_ds

# (4)数据可视化
def data_visualize(train_ds, is_augmentation = False):
    plt.figure(figsize = (10, 10))
    for images, labels in train_ds.take(1):
        for i in range(1):
            if is_augmentation:
                images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

# (5)数据增强(data augmentation)
def get_data_augmentation(image):
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    return data_augmentation(image)


if __name__ == "__main__":
    filter_out_corrupted_images()
    train_ds, validation_ds = get_dataset()

