 
TensorFlow SaveModel
=============================

为了将训练好的机器学习模型部署到各个目标平台(如服务器、移动端、嵌入式设备和浏览器等)，
我们的第一步往往是将训练好的整个模型完整导出(序列化)为一系列标准格式的文件。在此基础上，
我们才可以在不同的平台上使用相对应的部署工具来部署模型文件。

TensorFlow 提供了统一模型导出格式 ``SaveModel``, 这是我们在 TensorFlow 2 中主要使用的导出格式。
这样我们可以以这一格式为中介，将训练好的模型部署到多种平台上. 

同时，基于历史原因，Keras 的 Sequential 和 Functional 模式也有自有的模型导出格式。


0.tf.train.Checkpoint: 变量的保存与恢复
----------------------------------------

TensorFlow 的变量类型 ``ResourceVariable`` 并不能被序列化. 

TensorFlow 提供了 ``tf.train.Checkpoint`` 这一强大的变量保存与恢复类，使用它的 ``save()`` 和 ``restore()`` 方法可以保存
和恢复 TensorFlow 中的大部分对象。具体而言，``tf.keras.optimizer``，``tf.Variable``，``tf.keras.Layer`` 或者 ``tf.keras.Model``
实例都可以被保存，使用语法如下：

    .. code-block:: python

        checkpoint = tf.train.Checkpoint(model = model)

示例:

    .. code-block:: python

        # 保存模型及优化器
        checkpoint = tf.train.Checkpoint(myAwesomeModel = model, myAwesomeOptimizer = optimizer)
        # 保存模型
        checkpoint.save(save_path_with_prefix)

        # 载入模型
        model_to_be_restored = MyModel()
        checkpoint = tf.train.Checkpoint(myAwesomeModel = model_to_be_restored)
        checkpoint.restore(save_path_with_prefix_and_index)


1.使用 SaveModel 完整导出模型
----------------------------------------

作为模型导出格式的 ``SaveModel`` 包含了一个 TensorFlow 程序的完整信息: 不仅包含参数的权值，还包含计算的流程(计算图)。
当模型导出为 SaveModel 文件时，无须模型的源代码即可再次运行模型, 这使得 ``SaveModel`` 尤其适用于模型的分享和部署。

Keras 模型均可以方便地导出为 ``SaveModel`` 格式。不过需要注意的是，因为 ``SaveModel`` 基于计算图，
所以对于通过继承 ``tf.keras.Model`` 类建立的 Keras 模型来说，需要导出为 ``SaveModel`` 格式的方法(比如 call) 都需要
使用 ``@tf.function`` 修饰。


语法:

    .. code-block:: python

        # 保存
        tf.saved_model.save(model, "保存的目标文件夹名称")

        # 载入
        model = tf.saved_model.load("保存的目标文件夹名称")

示例:

    .. code-block:: python

        pass


2.Keras 自有的模型导出格式
----------------------------------------

示例:

    .. code-block:: shell

        curl -LO https://raw.githubcontent.com/keras-team/keras/master/examples/mnist_cnn.py


    .. code-block:: python

        model.save("mnist_cnn.h5")

    
    .. code-block:: python
    
        import keras

        keras.models.load_model("mnist_cnn.h5")