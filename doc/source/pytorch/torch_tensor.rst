
pytorch tensor
=================================================

内容概要
-------------------------------------------------

   - 1.tensor 的简介及创建

      - tensor 是多维数组
      - tensor 的创建

         - 直接创建

            - ``torch.tensor``
            - ``torch.from_numpy``
         
         - 依数值创建

            - ``empty``
            - ``ones``
            - ``zeros``
            - ``eye``
            - ``full``
            - ``arange``
            - ``linspace``

         - 依概率分布创建

            - ``torch.normal``
            - ``torch.randn``
            - ``torch.rand``
            - ``torch.randint``
            - ``torch.randperm``

   - 2.tensor 的操作

      - tensor 的基本操作

         - tensor 的拼接

            - ``torch.cat()``
            - ``torch.stack()``

         - tensor 的切分

            - ``torch.chunk()``
            - ``torch.split()``

         - tensor 的索引

            - ``index_select()``
            - ``masked_select()``

         - tensor 的变换

            - ``torch.reshape()``
            - ``torch.transpose()``
            - ``torch.t``

      - tensor 的数学运算

         - ``add(input, aplha, other)``

1.pytorch tensor 的创建
-------------------------------------------------

1.1 tensor 的介绍
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   tensor 是 pytorch 中最基本的概念,其参与了整个运算过程,这里主要介绍 tensor 的概念和属性,如 data, variable, device 等,
   并且介绍 tensor 的基本创建方法,如直接创建、依属主创建、依概率分布创建等.

      - tensor
      
         - tensor 其实是多维数组,它是标量、向量、矩阵的高维拓展

      - tensor 与 variable 

         - 在 pytorch 0.4.0 版本之后 variable 已经并入 tensor,但是 variable 这个数据类型对于理解 tensor 来说很有帮助,
           variable 是 ``torch.autograd`` 中的数据类型

         - variable(``torch.autograd.variable``) 有 5 个属性, 这些属性都是为了 tensor 的自动求导而设置的:

            - ``data``
            - ``grad``
            - ``grad_fn``
            - ``requires_grad``
            - ``is_leaf``

         - tensor(``torch.tensor``) 有 8 个属性：

            - 与数据本身相关

               - ``data``: 被包装的 tensor
               - ``dtype``：tensor 的数据类型,如 ``torch.floattensor``, ``torch.cuda.floattensor``, ``float32``, ``int64(torch.long)``
               - ``shape``: tensor 的形状
               - ``device``：tensor 所在的设备, gpu/cup,tensor 放在 gpu 上才能使用加速
            
            - 与梯度求导相关

               - ``requires_grad``: 是否需要梯度
               - ``grad``: ``data`` 的梯度
               - ``grad_fn``: fn 表示 function 的意思，记录创建 tensor 时用到的方法
               - ``is_leaf``: 是否是叶子节点(tensor)


1.2 tensor 的创建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: python

      from __future__ import print_function
      import numpy as np
      import torch


1.2.1 直接创建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.torch.tensor(): 从 data 创建 tensor api
   
   - API:

      .. code-block:: python

         torch.tensor(
            data,                   # list, numpy
            dtype = none,
            device = none,
            requires_grad = false,
            pin_memory = false      # 是否存于锁页内存
         )

   - 示例：

      .. code-block:: python

         arr = np.ones((3, 3))
         t = torch.tensor(arr, device = "cuda")
         print(t)

2. 通过 numpy array 来创建 tensor api

   - 创建的 tensor 与原 ndarray 共享内存，当修改其中一个数据的时候，另一个也会被改动

   - API:

      .. code-block:: python

         torch.from_numpy(ndarray)

   - 示例：

      .. code-block:: python

         arr = np.array([[1, 2, 3], [4, 5, 6]])
         t = torch.from_numpy(arr)
         print(arr)
         arr[0, 0] = 0
         print(arr, t)
         t[1, 1] = 100
         print(arr, t)

1.2.2 依数值创建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - api:

      .. code-block:: python

         torch.zeros(
            *size,
            out = none,             # 输出张量，就是把这个张量赋值给另一个张量，但这两个张量一样，指的是同一个内存地址
            dtype = none,
            layout = torch.strided, # 内存的布局形式
            device = none,
            requires_grad = false
         )

   - 示例：

   .. code-block:: python

      out_t = torch.tensor([1])
      t = torch.zeros((3, 3), out = out_t)
      print(out_t, t)
      print(id(out_t), id(t), id(t) == id(out_t))


2.pytorch tensor 的操作
--------------------------

   -  torch.add(, out)
   -  .add_()
   -  .view()

- add:

   .. code:: python

      x = torch.zeros(5, 3, dtype = torch.long)
      y = torch.rand(5, 3)

      # method 1
      print(x + y)

      # method 2
      print(torch.add(x, y))

      # method 3
      result = torch.empty(5, 3)
      torch.add(x, y, out = result)
      print(result)

      # method 4
      y.add_(x)
      print(y)

- index:

   .. code:: python

      x = torch.zeros(5, 3, dtype = torch.long)
      print(x[:, 1])

- resize:

   .. code:: python

      x = torch.randn(4, 4)
      y = x.view(16)
      z = x.view(-1, 8)
      print(x.size(), y.size(), z.size())

- object trans:

   .. code:: python

      x = torch.randn(1)
      print(x)
      print(x.item()) # python number

- torch tensor 2 numpy array:

   .. code:: python

      a = torch.ones(5)
      b = a.numpy()
      print(a)
      print(b)

      a.add_(1)
      print(a)
      print(b)

- numpy array 2 torch tensor:

   .. code:: python

      import numpy as np
      a = np.ones(5)
      b = torch.from_numpy(a)
      np.add(a, 1, out = a)
      print(a)
      print(b)

.. _header-n50:

3.pytorch cuda tensor
--------------------------

   .. code:: python

      # let us run this cell only if cuda is available
      # we will use ``torch.device`` objects to move tensors in and out of gpu
      if torch.cuda.is_available():
         device = torch.device("cuda")          # a cuda device object
         y = torch.ones_like(x, device=device)  # directly create a tensor on gpu
         x = x.to(device)                       # or just use strings ``.to("cuda")``
         z = x + y
         print(z)
         print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
