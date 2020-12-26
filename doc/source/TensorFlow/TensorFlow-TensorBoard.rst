
TensorFlow TensorBoard
==========================

1.实时查看参数变化情况
------------------------------------

1.1 TensorBoard 使用介绍
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   1.首先，在代码目录下建立一个文件夹，存放 TensorBoard 的记录文件

      .. code-block:: shell

         $ mkdir tensorboard

   2.在代码中实例化一个记录器

      .. code-block:: python

         summary_writer =  tf.summary.create_file_writer("./tensorboard")

   3.当需要记录训练过程中的参数时，通过 ``with`` 语句指定希望使用的记录器，并对需要记录的参数(一般是标量)运行:

      .. code-block:: python
         
         with summary_writer.as_default():
            tf.summary.scalar(name, tensor, step = batch_index)

   4.当要对训练过程可视化时，在代码目录打开终端

      .. code-block:: shell

         $ tensorboard --logdir=./tensorboard

   5.使用浏览器访问命令行程序所输出的网址, 即可访问 TensorBoard 的可视化界面

      - ``http://计算机名称:6006``

.. note:: 

   - 每运行一次 ``tf.summary.scalar()``，记录器就会向记录文件中写入一条记录
   - 除了最简单的标量以外，TensorBoard 还可以对其他类型的数据，如：图像、音频等进行可视化
   - 默认情况下，TensorBoard 每 30 秒更新一次数据，可以点击右上角的刷新按钮手动刷新
   - TensorBoard 的使用有以下注意事项:
      - 如果需要重新训练，那么删除掉记录文件夹内的信息并重启 TensorBoard，
        或者建立一个新的记录文件夹并开启 TensorBoard，将 ``--logdir`` 参数设置为新建里的文件夹
      - 记录文件夹目录许保持全英文

1.2 TensorBoard 代码框架
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python
      
      # (1)实例化一个记录器
      summary_writer =  tf.summary.create_file_writer("./tensorboard")
      
      # (2)开始训练模型
      for batch_index in range(num_batches):
         # ...(训练代码，将当前 batch 的损失值放入变量 loss 中)

         # (3)指定记录器
         with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step = batch_index)
            tf.summary.scalar("MyScalar", my_scalar, step = batch_index)

2.查看 Graph 和 Profile 信息
-------------------------------------

   在训练时使用 ``tf.summary.trace_on`` 开启 Trace，此时 TensorFlow 会将训练时的大量信息，
   如：计算图的结构、每个操作所耗费的时间等，记录下来。

   在训练完成后，使用 ``tf.summary.trace_export`` 将记录结果输出到文件。


   1.使用 TensorBoard 代码框架对模型信息进行跟踪记录

      .. code-block:: python

         # (1)实例化一个记录器
         summary_writer =  tf.summary.create_file_writer("./tensorboard")

         # (2)开启 Trace, 可以记录图结构和 profile 信息
         tf.summary.trace_on(graph = True, profiler = True)
         
         # (3)开始训练模型
         for batch_index in range(num_batches):
            # (4)...(训练代码，将当前 batch 的损失值放入变量 loss 中)
            
            # (5)指定记录器, 将当前指标值写入记录器
            with summary_writer.as_default():
               tf.summary.scalar("loss", loss, step = batch_index)
               tf.summary.scalar("MyScalar", my_scalar, step = batch_index)
         
         # (6)保存 Trace 信息到文件
         with summary_writer.as_default():
            tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = log_dir)

   2.在 TensorBoard 的菜单中选择 ``PROFILE``，以时间轴方式查看各操作的耗时情况，
   如果使用了 ``@tf.function`` 建立计算图，也可以点击 ``GRAPHS`` 查看图结构
