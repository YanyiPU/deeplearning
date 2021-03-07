.. _header-n0:

torch save model
================

.. _header-n4:

1.PyTorch 保存模型
------------------

.. code:: python

   import torch

.. _header-n6:

1.1 方法 1：保存和加载模型参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 保存模型
   torch.save(the_model.state_dict(), PATH)

   # 加载模型
   the_model = TheModelClass(*args, **kwargs)
   the_model.load_state_dict(torch.load(PATH))

.. _header-n8:

1.2 方法 2：保存和加载整个模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 保存模型
   torch.save(the_model, PATH)

   # 加载模型
   the_model = torch.load(PATH)
