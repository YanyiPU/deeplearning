
TensorFlow Env
======================

1.TensorFlow 支持的硬件平台
----------------------------

    - PC

        - CPU

        - GPU

        - Cloud TPU(张量)

    - Mobile

        - Android 

        - iOS

        - Embedded Devices


2.TensorFlow 系统要求
-----------------------------------------

    - Python 3.5-3.7

    - pip >= 19.0(需要 manylinux2010 支持)

    - Ubuntu 16.04 or later(64位)

    - Windows 7 or later(64位, 仅支持 Python 3)

    - macOS 10.12.6(Sierra) or later(64位, no GPU support)

    - Raspbian 9.0 or later

    - GPU 支持需要使用支持 CUDA® 的显卡（适用于 Ubuntu 和 Windows）


3.安装 TensorFlow 2
---------------------------

.. important:: TensorFlow pip packages

    - tensorflow: 支持 CPU 和 GPU 的最新稳定版（适用于 Ubuntu 和 Windows）

    - tf-nightly: 预览 build（不稳定）。Ubuntu 和 Windows 均包含 GPU 支持

    - tensorflow-gpu: Current release with GPU support (Ubuntu and Windows)

    - tf-nightly-gpu: Nightly build with GPU support (unstable, Ubuntu and Windows)

3.1 使用 pip 安装 TensorFlow 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    - 必须使用最新版本的 pip, 才能安装 TensorFlow 2

- Virtualenv 安装 

    .. code-block:: shell
        
        # Requires the latest pip
        (venv) $ pip install --upgrade pip
        
        # Current stable release for CPU and GPU
        (venv) $ pip install --upgrade tensorflow
        (venv) $ python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

- 系统安装

    .. code-block:: shell
    
        # Requires the latest pip
        $ pip3 install --upgrade pip

        # Current stable release for CPU and GPU
        $ pip3 install --user --upgrade tensorflow
        $ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"


3.2 使用 Docker 安装 TensorFlow 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - TensorFlow Docker 映像已经过配置，可运行 TensorFlow。Docker 容器可在虚拟环境中运行，是设置 GPU 支持的最简单方法。

        .. code-block:: shell
        
            # Download latest stable image
            $ docker pull tensorflow/tensorflow:latest-py3

            # Start Jupyter server
            $ docker run -it -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter

.. note:: TensorFlow Docker Image

    - https://hub.docker.com/r/tensorflow/tensorflow/


3.3 Google Colab
~~~~~~~~~~~~~~~~~~

    - https://colab.research.google.com/notebooks/welcome.ipynb?hl=zh_cn