.. _header-n0:

Torch
=====

A typical training procedure for a neural network is as follows:

-  Define the neural network that has some learnable parameters (or
   weights)

-  Iterate over a dataset of inputs

-  Process input through the network

-  Compute the loss (how far is the output from being correct)

-  Propagate gradients back into the network’s parameters

-  Update the weights of the network, typically using a simple update
   rule: ``weight = weight - learning_rate * gradient``

.. _header-n18:

定义神经网络
------------

.. code:: python

   # nn
   # autograd
   # nn.Module
       # forward(input) => output

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class Net(nn.Module):

       def __init__(self):
           super(Net, self).__init__()
           # 1 input image channel, 6 output channels, 3x3 square convolution kernel
           self.conv1 = nn.Conv2d(1, 6, 3)
           self.conv2 = nn.Conv2d(6, 16, 3)
           # an affine operation: y = Wx + b
           self.fc1 = nn.Linear(16 * 6 * 6, 120)
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)

       def forward(self, x):
           # Max pooling over a (2, 2) window
           x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
           # If the size is a square you can only specify a single number
           x = F.max_pool2d(F.relu(self.conv2(x)), 2)
           x = x.view(-1, self.num_flat_features(x))
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)

           return x

       def num_flat_features(self, x):
           size = x.size()[1:]
           num_features = 1
           for s in size:
               num_features *= s

           return num_features


   net = Net()
   print(net)

.. code:: python

   params = list(net.parameters)
   print(len(params))
   print(params[0].size()) # conv1's .weight

.. code:: python

   net.zero_grad()
   out.backward(torch.randn(1, 10))
