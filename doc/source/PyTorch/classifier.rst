.. _header-n0:

训练一个图像分类器
==================

.. _header-n3:

Data
----

-  图片

   -  ``Pillow``

   -  ``OpenCV``

   -  ``torchvision``

-  语音

   -  ``scipy``

   -  ``librosa``

-  文本

   -  ``Python``

   -  ``Cython``

   -  ``NLTK``

   -  ``SpaCy``

.. _header-n32:

加载并正规化数据
----------------

加载训练数据、测试数据：

.. code:: python

   import torch
   import torchvision
   import torchvision.transforms as transforms


   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   train_set = torchvision.datasets.CIFAR10(root = "./data", 
                                            train = True,
                                            download = True, 
                                            transform = transform)
   train_loader = torch.utils.data.DataLoader(train_set, 
                                              batch_size = 4, 
                                              shuffle = True, 
                                              num_workers = 2)
   test_set = torchvision.datasets.CIFAR10(root = "./data", 
                                           train = False 
                                           download = True, 
                                           transform = transform)
   test_loader = torch.utils.data.DataLoader(test_set, 
                                             batch_size = 4, 
                                             shuffle = False, 
                                             num_workers = 2)
   classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

查看训练数据：

.. code:: python

   import matplotlib.pyplot as plt 
   import numpy as np 

   def imshow(img):
       img = img / 2 + 0.5 
       npimg = img.numpy()
       plt.imshow(np.transpose(npimg, (1, 2, 0)))
       plt.show()

   dataiter = iter(train_loader)
   images, labels = dataiter.next()

   imshow(torchvision.utils.make_grid(images))
   print(" ".join("%5s" % classes[labels[j]] for j in range(4)))

.. _header-n37:

定义卷积神经网络
----------------

.. code:: python

   import torch.nn as nn
   import torch.nn.function as F

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(3, 6, 5)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.Conv2d(6, 16, 5)
           self.fc1 = nn.Linear(16 * 5 * 5, 120)
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)

       def forward(self, x):
           x = self.pool(F.relu(self.conv1(x)))
           x = self.pool(F.relu(self.conv2(x)))
           x = x.view(-1, 16 * 5 * 5)
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)
           return x

   net = New()

.. _header-n40:

定义损失函数和优化器
--------------------

.. code:: python

   import torch.optim as optim

   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

.. _header-n42:

训练网络
--------

.. code:: python

   for epoch in range(2): # loop over the dataset multiple times
       running_loss = 0.0
       for i, data in enumerate(train_loader, 0):
           # get the inputs; data is a list of [inputs, labels]
           inputs, labels = data
           optimizer.zero_grad()

           # forward + backward + optimize
           outputs = net(inputs)
           loss = criterion(outputs, lables)
           loss.backward()
           optimizer.step()

           # print statistics
           running_loss += loss.item()
           if i % 2000 == 1999:
               print("[%d, %d] loss: %.3f" % 
                     (epoch + 1, i + 1, running_loss / 2000))
               running_loss = 0.0


   print("Finished Training.")

.. code:: python

   PATH = "./cifar_net.pth"
   torch.save(net.state_dict(), PATH)

.. _header-n45:

测试神经网路
------------

.. code:: python

   dataiter = iter(testloader)
   images, labels = dataiter.next()

   # print image
   imshow(torchvision.utils.make_gird(images))
   print("GroundTruth: ", " ".join("%5s" % 
         classes[labels[j]] for j in range(4)))

.. code:: python

   new = New()
   net.load_state_dict(torch.load(PATH))

.. code:: python

   outputs = net(images)

.. code:: python

   _, predicted = torch.max(outputs, 1)
   print("Predicted: ", " ".join("%5s" % 
         sclasses[predicted[j]] for j in range(4)))

.. code:: python

   correct = 0
   total = 0
   with torch.no_grad():
       for data in test_loader:
           images, labels = data
           outputs = net(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

   print("Accuracy of the network on the 10000 test images: %d %%" % 
         (100 * correct / total))

.. code:: python

   class_correct = list(0. for i in range(10))
   class_total = list(0. for i in range(10))
   with torch.no_grad():
       for data in testloader:
           images, labels = data
           outputs = net(images)
           _, predicted = torch.max(outputs, 1)
           c = (predicted == labels).squeeze()
           for i in range(4):
               labels = labels[i]
               class_correct[labels] += c[i].item
               class_total[labels] += 1

   for i in range(10):
       print("Accuracy of %5s : %2d %%" % 
             (classes[i], 100 * class_correct[i] / class_total[i]))

.. _header-n53:

在GPU上训练模型
---------------

.. code:: python

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   print(device)
   net.to(device)
   inputs, labels = data[0].to(device), data[1].to(device)
