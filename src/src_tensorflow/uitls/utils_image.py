# -*- coding: utf-8 -*-
import os
import pathlib
import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def image_download(data_url, data_fname, untar = False):
    """
    下载图片数据
    Args:
        data_url: 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
        data_fname:
        untar:
    """
    data_root_origin = tf.keras.utils.get_file(
        origin = data_url,
        fname = data_fname,
        untar = untar
    )
    data_root = pathlib.Path(data_root_origin)
    for item in data_root.iterdir():
        print(item)
    
    return data_root


def image_paths(data_root, is_shuffle = False):
    """
    获取所有图像地址
    """
    all_image_paths = list(data_root.glob("*/*"))
    all_image_paths = [str(path) for path in all_image_paths]
    if is_shuffle:
        random.shuffle(all_image_paths)
    # 图像数量
    image_count = len(all_image_paths)
    print(f"image_count={image_count}")

    return all_image_paths


def image_load_preprocessing(img_path):
    """
    加载和格式化图像数据
    Args:
        img_path:
        size:
    """
    img_raw = tf.io.read_file(img_path)
    # img_tensor = tf.image.decode_image(img_raw, channels = 3)
    img_tensor = tf.image.decode_jpeg(img_raw, channels = 3)
    # img_tensor = tf.image.decode_and_crop_jpeg()
    # img_tensor = tf.image.decode_bmp()
    # img_tensor = tf.image.decode_gif()
    # img_tensor = tf.image.decode_png()
    img_resized = tf.image.resize(img_tensor, [192, 192])
    image = img_resized / 255.0

    return image


def image_preprocessing(img_path):
    """
    加载和格式化图像数据
    Args:
        img_path:
    """
    # img_raw = tf.io.read_file(img_path)
    # img_tensor = tf.image.decode_image(img_raw, channels = 3)
    img_tensor = tf.image.decode_jpeg(img_raw, channels = 3)
    # img_tensor = tf.image.decode_and_crop_jpeg()
    # img_tensor = tf.image.decode_bmp()
    # img_tensor = tf.image.decode_gif()
    # img_tensor = tf.image.decode_png()
    img_resized = tf.image.resize(img_tensor, [192, 192])
    image = img_resized / 255.0

    return image



def image_display(img_path):
    """
    展示图片
    """
    display.display(display.Image(img_path))


def image_labels(data_root, all_image_paths):
    all_label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
    all_label_dict =  dict((name, index) for index, name in enumerate(all_label_names))
    all_image_labels = [all_label_dict[pathlib.Path(path).parent.name] for path in all_image_paths]

    return all_label_names, all_label_dict, all_image_labels


def image_display_v2(img_path, img_label, all_label_names):
    image = image_load_preprocessing(img_path)
    plt.imshow(image)
    plt.grid(False)
    # plt.xlabel()
    plt.title(all_label_names[img_label].title())

    
    
# data_augmentation = tf.keras.models.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal", input_shape = (img_height, img_width, 3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
#     layers.experimental.preprocessing.RandomZoom(0.1),
# ])