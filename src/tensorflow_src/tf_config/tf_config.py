import tensorflow as tf


"""
TensorFlow 提供了 tf.config 模块来帮助设置 GPU 的使用和分配方式

    1.当出现多人共同使用一台多 GPU 的工作站的时候，默认情况下，TensorFlow 会使用其所能够使用的所有 GPU，
      这时就需要合理分配显卡资源。
"""


# -----------------------
# 1.获得当前主机上某种特定运算设备类型(GPU,CPU)的列表
# -----------------------
gpus = tf.config.list_physical_devices(device_type = "GPU")
cpus = tf.config.list_physical_devices(device_type = "CPU")
print(gpus, cpus)


# -----------------------
# 2.设置当前程序可见的设备范围(当前程序只会使用自己可见的设备，不可见的设备不会被当前程序使用)
# (1) tf.config.set_visible_devices
# (2) CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU
#     $ export CUDA_VISIBLE_DEVICES=2,3
# -----------------------
# tf.config.set_visible_devices
gpus = tf.config.list_physical_devices(device_type = "GPU")
tf.config.set_visible_devices(devices = gpus[0:2], device_type = "GPU")

# CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU
import os
os.environ("CUDA_VISIBLE_DEVICES") = "2,3"

# -----------------------
# 3.设置显存使用策略
# (1)仅在需要时申请显存空间
# (2)限制消耗固定大小的显存
# -----------------------
# 仅在需要时申请显存空间
gpus = tf.config.list_physical_devices(device_type = "GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device = gpu, enable = True)

# 限制消耗固定大小的显存为 1GB
gpus = tf.config.list_physical_devices(device_type = "GPU")
tf.config.set_virtual_device_configuration(
    gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 1024)]
)

# -----------------------
# 4.单 GPU 模拟多 GPU 环境
# -----------------------
# 在实体 GPU (GPU:0) 的基础上建立两个显存均为2GB的虚拟 GPU
gpus = tf.config.list_physical_devices(device_type = "GPU")
tf.config.set_virtual_device_configuration(
    gpus[0],
    [tf.config.VirtualDeviceConfiguration(memory_limit = 2048),
     tf.config.VirtualDeviceConfiguration(memory_limit = 2048),]
)
