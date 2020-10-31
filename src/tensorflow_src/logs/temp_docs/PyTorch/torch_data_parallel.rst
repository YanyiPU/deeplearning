.. _header-n0:

PyTorch 数据并行
=================

.. _header-n3:

让模型跑在 GPU 上
-----------------

.. code:: python

   import torch

   # 让模型在 GPU 上运行
   device = torch.device("cuda:0")
   model.to(device)

   # 将 tensor 复制到 GPU 上
   my_tensor = torch.ones(2, 2, dtype = torch.double)
   mytensor = my_tensor.to(device)

.. _header-n5:

让模型跑在多个 GPU 上
---------------------

-  PyTorch 默认使用单个 GPU 执行运算

.. code:: python

   model = nn.DataParallel(model)

.. code:: python

   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader

   # Parameters and DataLoaders
   input_size = 5
   output_size = 2

   batch_size = 30
   data_size = 100

   # Device
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

.. code:: python

   class RandomDataset(Dataset):

   	def __init__(self, size, length):
   		self.len = length
   		self.data = torch.randn(length, size)

   	def __getitem__(self, index):
   		return self.data[index]

   	def __len__(self):
   		return self.len

   rand_loader = DataLoader(dataset = RandomDataset(input_size, data_size), 
   						 batch_size = batch_size, 
   						 shuffle = True)

.. code:: python

   class Model(nn.Module):

   	def __init__(self, input_size, output_size):
   		super(Model, self)__init__()
   		self.fc = nn.Linear(input_size, output_size)

   	def forward(self, input):
   		output = self.fc(input)
   		print("\tIn Model: input size", input.size(),
   			  "output size", output.size())

   		return output

.. code:: python

   model = Model(input_size, output_size)
   if torch.cuda.device_count() > 1:
   	print("Let's use", torch.cuda.device_count(), "GPUs!")
   	model = nn.DataParallel(model)

   model.to(device)

.. code:: python

   for data in rand_loader:
   	input = data.to(device)
   	output = model(input)
   	print("Outside: input size", input.size(),
   		  "output_size", output.size())
