

PyTorch Cheat Sheet
=======================

1.Imports
-----------------------

   .. code-block:: python

      # General
      import torch # root package
      from torch.utils.data import Dataset, Dataloader # dataset representation and loading

      # Neural Network API
      from torch import Tensor
      import torch.nn as nn
      import torch.nn.functional as F
      import torch.autograd as autograd
      import torch.optim as optim
      from torch.jit import script, trace

      # Torchscript and JIT
      torch.jit.trace()
      @script

      # ONNX
      torch.onnx.export(model, dummy data, xxxx.proto)
      model = onnx.load("alexnet.proto)
      onnx.checker.check_model(model)
      onnx.helper.printable_graph(model.graph)

      # Vision
      from torchvision import datasets, models, transforms
      import torchvision.transforms as transforms

      # Distributed Training
      import torch.distributed as dist
      from torch.multiprocessing import Process


2.Tensors
----------------------------

   .. code-block:: python
   
      import torch

      # Creation
      x = torch.randn(*size)
      x = torch.[ones|zeros](*size)
      x = torch.tensor(L)
      y = x.clone()
      with torch.no_grad():
      requires_grad = True

      # Dimensionality
      x.size()
      x = torch.cat(tensor_seq, dim = 0)
      y = x.view(a, b, ...)
      y - x.view(-1,a)
      y = x.transpose(a, b)
      y = x.permute(*dims)
      y = x.unsqueeze(dim)
      y = x.unsqueeze(dim = 2)
      y = x.squeeze()
      y = x.squeeze(dim = 1)

      # Algebra
      ret = A.mm(B)
      ret = A.mv(x)
      x = x.t()

      # GUP Usage
      torch.cuda.is_available
      x = x.cuda()
      x = x.cpu()
      if not args.disable_cuda and torch.cuda.is_available():
         args.device = torch.device("cuda")
      else:
         args.device = torch.device("cpu")
      net.to(device)
      x = x.to(device)

3.Deep Learning
-------------------------------

   .. code-block:: python

      import torch.nn as nn
      
      nn.Linear(m, n)
      nn.ConvXd(m, n, s)
      nn.MaxPoolXd(s)
      nn.BatchNormXd
      nn.RNN
      nn.LSTM
      nn.GRU
      nn.Dropout(p = 0.5, inplace = False)
      nn.Dropout2d(p = 0.5, inplace = False)
      nn.Embedding(num_embeddings, embedding_dim)

      # Loss Function
      nn.X

      # Activation Function
      nn.X

      # Optimizers
      opt = optim.x(model.parameters(), ...)
      opt.step()
      optim.X

      # Learning rate scheduling
      scheduler = optim.X(optimizer, ...)
      scheduler.step()
      optim.lr_scheduler.X

4.Data Utilities
--------------------------------

   .. code-block:: python

      # Dataset
      Dataset
      TensorDataset
      Concat Dataset

      # Dataloaders and DataSamplers
      DataLoader(dataset, batch_size = 1, ...)
      sampler.Sampler(dataset, ...)
      sampler.XSampler where ...
