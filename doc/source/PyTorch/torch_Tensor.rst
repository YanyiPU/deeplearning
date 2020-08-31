
PyTorch Tensor
======================

内容概要
-------------------------------------------------

   - 1.Tensor 的简介及创建

      - Tensor 是多维数组

      - Tensor 的创建

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

   - 2.Tensor 的操作

      - Tensor 的基本操作

         - Tensor 的拼接

            - ``torch.cat()``
            - ``torch.stack()``

         - Tensor 的切分

            - ``torch.chunk()``
            - ``torch.split()``

         - Tensor 的索引

            - ``index_select()``
            - ``masked_select()``

         - Tensor 的变换

            - ``torch.reshape()``
            - ``torch.transpose()``
            - ``torch.T``

      - Tensor 的数学运算

         - ``add(input, aplha, other)``

1.PyTorch Tensor 的创建
-------------------------------------------------

1.1 Tensor 的介绍
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor 是 PyTorch 中最基本的概念,其参与了整个运算过程,这里主要介绍 Tensor 的概念和属性,如 data, device, device 等,
并且介绍 Tensor 的基本创建方法,如直接创建、依属主创建、依概率分布创建等.

   - Tensor
   
      - Tensor 其实是多维数组,它是标量、向量、矩阵的高维拓展

   - Tensor 与 Variable 

      - 在 Pytorch 0.4.0 版本之后 Variable 已经并入 Tensor,但是 Variable 这个数据类型对于理解 Tensor 来说很有帮助,
        Variable 是 ``torch.autograd`` 中的数据类型

      - Variable(``torch.autograd.Variable``) 有 5 个属性, 这些属性都是为了 Tensor 的自动求导而设置的:

         - ``data``
         - ``grad``
         - ``grad_fn``
         - ``requires_grad``
         - ``is_leaf``

      - Tensor(``torch.Tensor``) 有 8 个属性：

         - 与数据本身相关

            - ``data``: 被包装的 Tensor
            - ``dtype``：Tensor 的数据类型,如 ``torch.FloatTensor``, ``torch.cuda.FloatTensor``, ``float32``, ``int64(torch.long)``
            - ``shape``: Tensor 的形状
            - ``device``：Tensor 所在的设备, GPU/CUP,Tensor 放在 GPU 上才能使用加速
         
         - 与梯度求导相关

            - ``requires_grad``: 是否需要梯度
            - ``grad``: ``data`` 的梯度
            - ``grad_fn``: fn 表示 function 的意思，记录创建 Tensor 时用到的方法
            - ``is_leaf``: 是否是叶子节点(Tensor)


1.2 Tensor 的创建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: python

      from __future__ import print_function
      import numpy as np
      import torch


直接创建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

torch.tensor(): 从 data 创建 Tensor API
''''''''''''''''''''''''''''''''''''''''''''''''''''

- API:

   .. code-block:: python

      torch.tensor(
         data,                   # list, numpy
         dtype = None,
         device = None,
         requires_grad = False,
         pin_memory = False      # 是否存于锁页内存
      )

- 示例：

   .. code-block:: python

      arr = np.ones((3, 3))
      t = torch.tensor(arr, device = "cuda")
      print(t)

通过 numpy array 来创建 Tensor API
''''''''''''''''''''''''''''''''''''''''''''''''''''

创建的 Tensor 与原 ndarray 共享内存，当修改其中一个数据的时候，另一个也会被改动

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

依数值创建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- API:

.. code-block:: python

   torch.zeros(
      *size,
      out = None,             # 输出张量，就是把这个张量赋值给另一个张量，但这两个张量一样，指的是同一个内存地址
      dtype = None,
      layout = torch.strided, # 内存的布局形式
      device = None,
      requires_grad = False
   )

- 示例：

.. code-block:: python

   out_t = torch.tensor([1])
   t = torch.zeros((3, 3), out = out_t)
   print(out_t, t)
   print(id(out_t), id(t), id(t) == id(out_t))


2.PyTorch Tensor 的操作
--------------------------

-  torch.add(, out)

-  .add_()

-  .view()

Add:

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

Index:

.. code:: python

   x = torch.zeros(5, 3, dtype = torch.long)
   print(x[:, 1])

Resize:

.. code:: python

   x = torch.randn(4, 4)
   y = x.view(16)
   z = x.view(-1, 8)
   print(x.size(), y.size(), z.size())

object trans:

.. code:: python

   x = torch.randn(1)
   print(x)
   print(x.item()) # Python number

Torch Tensor 2 Numpy Array:

.. code:: python

   a = torch.ones(5)
   b = a.numpy()
   print(a)
   print(b)

   a.add_(1)
   print(a)
   print(b)

Numpy Array 2 Torch Tensor:

.. code:: python

   import numpy as np
   a = np.ones(5)
   b = torch.from_numpy(a)
   np.add(a, 1, out = a)
   print(a)
   print(b)

.. _header-n50:

3.PyTorch CUDA Tensor
--------------------------

.. code:: python

   # let us run this cell only if CUDA is available
   # We will use ``torch.device`` objects to move tensors in and out of GPU
   if torch.cuda.is_available():
       device = torch.device("cuda")          # a CUDA device object
       y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
       x = x.to(device)                       # or just use strings ``.to("cuda")``
       z = x + y
       print(z)
       print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

.. _header-n52:






