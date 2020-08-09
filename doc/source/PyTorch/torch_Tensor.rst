.. _header-n0:

Torch Tensor
============

.. _header-n3:

What is PyTorch?
----------------

-  A replacement for Numpy to use the power of GPUs

-  A deep learning research platform that provides maximum flexibility
   and speed

.. code:: python

   from __future__ import print_function
   import torch

.. _header-n10:

1.Tensors
---------

Tensors

-  torch.empyt(())

-  torch.rand()

-  torch.zeros(, dtype)

-  torch.tensor([])

-  .new_ones(, dtype)

-  torch.randn_like(, dtype)

-  .dtype

-  .size()

.. code:: python

   x = torch.empty(5, 3)
   print(x)

   x = torch.rand(5, 3)
   print(x)

   x = torch.zeros(5, 3, dtype = torch.long)
   print(x)

   x = torch.tensor([5.5, 3])
   print(x)

   x = x.new_ones(5, 3, dtype = torch.double)
   print(x)

   x = torch.randn_like(x, dtype = torch.float)
   print(x)
   print(x.size())

.. _header-n30:

2.Operations
------------

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

3.CUDA Tensors
--------------

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

4.AutoGrad: automatic differentiation
-------------------------------------

-  package ``autograd``

-  torch.Tensor

-  .requires_grad = True

-  .backward()

-  .grad

-  .detach()

-  with torch.no_grad(): pass

-  .grad_fn
