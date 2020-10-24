.. _header-n0:

weight\ *decay*\ sample
=======================

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-
   # @Date    : 2019-08-04 13:51:19
   # @Author  : Your Name (you@example.org)
   # @Link    : http://example.org
   # @Version : $Id$


   import d2lzh as d2l
   from mxnet import autograd, gluon, init, nd
   from mxnet.gluon import data as gdata, loss as gloss, nn


   # #################################################################
   # 构造模拟数据集
   # #################################################################
   n_train, n_test, num_inputs = 20, 100, 200
   true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

   features = nd.random.normal(shape = (n_train + n_test, num_inputs))
   labels = nd.dot(features, true_w) + true_b
   labels += nd.random.normal(scale = 0.01, shape = labels.shape)

   train_features, test_features = features[:n_train, :], features[n_train:, :]
   train_labels, test_labels = labels[:n_train], lables[n_train:]

   # #################################################################
   # 初始化模型参数
   # #################################################################
   def init_params():
       w = nd.random.normal(scale = 1, shape = (num_inputs, 1))\
       b = nd.zeros(shape = (1, ))
       w.attach_grad()
       b.attach_grad()
       return [w, b]


   # #################################################################
   # 定义L2范数惩罚项
   # #################################################################
   def l2_penalty(w):
       penalty = w ** 2.sum() / 2
       return penalty

   # #################################################################
   # 定义训练和测试
   # #################################################################
   batch_size = 1
   num_epochs = 100
   lr = 0.003
   net = d2l.linreg
   loss = d2l.squared_loss
   train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),
                                 batch_size,
                                 shuffle = True)

   def fit_and_plot(wd):
   	net = nn.Sequential()
   	net.add(nn.Dense(1))
   	net.initialize(init.Normal(sigma = 1))
   	# 对权重参数衰减。权重名称一般是以weight结尾
   	train_w = gluon.Trainer(net.collect_params(".*weight"), "sgd", {"learning_rate": lr, "wd": wd})
   	# 不对偏差参数衰减。偏差名称一般是以bias结尾
   	trainer_b = gluon.Trainer(net.collect_params(".*bias"), "sgd", {"learning_rate": lr})

   	train_ls, test_ls = [], []
   	for _ in range(num_epochs):
   		for X, y in train_iter:
   			with autograd.record():
   				l = loss(net(X), y)
   			l.backward()
   			# 对两个Trainer实例分别调用step函数，从而分别更新权重和偏差
   			train_w.step(batch_size)
   			trainer_b.step(batch_size)
   		train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
   		test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
   	d2l.semilogy(range(1, num_epochs + 1), train_ls, "epochs", "loss",
   				 range(1, num_epochs + 1), test_ls, ["train", "test"])
   	print("L2 norm of w:", net[0].weight.data().norm().asscalar())


   fit_and_plot_gluon(0)
   fit_and_plot_gluon(3)
