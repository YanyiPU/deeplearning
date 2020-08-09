.. _header-n0:

PyTorch 自动求导
================

   ``autograd`` 包提供了对所有 Tensor 的自动微分操作

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-

   """
   `autograd` package:
   -------------------
   跟踪 torch.Tensor 上所有的操作：torch.Tensor(requires_grad = True)
   自动计算所有的梯度：.backward()
   torch.Tensor 上的梯度：.grad
   torch.Tensor 是否被跟踪：.requires_grad
   停止跟踪 torch.Tensor 上的跟踪历史、未来的跟踪：.detach()

   with torch.no_grad():
       pass

   Function
   .grad_fn
   """

   import torch

   # --------------------
   # 创建 Tensor 时设置 requires_grad 跟踪前向计算
   # --------------------
   x = torch.ones(2, 2, requires_grad = True)
   print("x:", x)

   y = x + 2
   print("y:", y)
   print("y.grad_fn:", y.grad_fn)

   z = y * y * 3
   print("z:", z)
   print("z.grad_fn", z.grad_fn)

   out = z.mean()
   print("out:", out)
   print("out.grad_fn:", out.grad_fn)
   out.backward()
   print("x.grad:", x.grad)

   # --------------------
   # .requires_grad_() 能够改变一个已经存在的 Tensor 的 `requires_grad`
   # --------------------
   a = torch.randn(2, 2)
   a = ((a * 3) / (a - 1))
   print("a.requires_grad", a.requires_grad)
   a.requires_grad_(True)
   print("a.requires_grad:", a.requires_grad)
   b = (a * a).sum()
   print("b.grad_fn", b.grad_fn)



   # 梯度
   x = torch.randn(3, requires_grad = True)
   y = x * 2
   while y.data.norm() < 1000:
       y = y * 2
   v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
   y.backward(v)
   print(x.grad)

   # .requires_grad
   print(x.requires_grad)
   print((x ** 2).requires_grad)

   with torch.no_grad():
       print((x ** 2).requires_grad)

   # .detach()
   print(x.requires_grad)
   y = x.detach()
   print(y.requires_grad)
   print(x.eq(y).all())
