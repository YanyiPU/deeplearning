.. _header-n0:

TensorFlow-sign-DNN
===================

步骤：

-  定义网络结构

   -  指定输入层、隐藏层、输出层的大小

-  初始化模型参数

-  循环操作：执行前向传播 => 计算当前损失 => 执行反向传播 => 权值更新

   -  执行前向传播

   -  计算当前损失

   -  执行反向传播

   -  权值更新

总结：

-  Tensorflow 语法中两个基本的对象类是 Tensor 和 Operator.

-  Tensorflow 执行计算的基本步骤为

   -  创建计算图（张量、变量和占位符变量等）

   -  创建会话

   -  初始化会话

   -  在计算图中执行会话

.. _header-n39:

data
----

.. image:: ../../../images/shoushi.png
   :alt: 

.. code:: python

   import numpy as np
   import tensorflow as tf

.. code:: python

   x_train_origin, y_train_origin, x_test_origin, y_test_origin, classes = load_dataset()

   # Flatten the training and test images
   x_train_flatten = x_train_origin.reshape(x_train_origin.shape[0], -1).T
   x_test_flatten = x_test_origin.reshape(x_test_origin.shape[0], -1)
   # Normalize image vectors 
   x_train = x_train_flatten / 255.0
   x_test = x_test_flatten / 255.0

   # Convert training and test labels to one hot matrices
   y_train = convert_to_one_hot(y_train_origin, 6)
   y_test = convert_to_one_hot(y_test_origin, 6)

   print ("number of training examples = " + str(x_train.shape[1]))
   print ("number of test examples = " + str(x_test.shape[1]))
   print ("x_train shape: " + str(x_train.shape))
   print ("y_train shape: " + str(y_train.shape))
   print ("x_test shape: " + str(x_test.shape))
   print ("y_test shape: " + str(y_test.shape))

.. _header-n43:

1.定义神经网络结构
------------------

   -  Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax

.. code:: python

   def create_placeholders(n_x, n_y):
       X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
       Y = tf.placeholder(tf.float32, shape = (n_y, None), name = "Y")
       return X, Y

.. _header-n49:

2.初始化模型参数
----------------

-  :math:`Input = X_{(12288, 1080)}`

-  :math:`H_{(25, 1080)}^{[1]} = relu\Big(W_{(25, 12288)}^{[1]} \times X_{(12288, 1080)} + b_{(25, 1)}^{[1]}\Big)`

-  :math:`H_{(12, 1080)}^{[2]} = relu\Big(W_{(12, 25)}^{[2]} \times H_{(25, 1080)}^{[1]} + b_{(12, 1)}^{[2]}\Big)`

-  :math:`H_{(6, 1080)}^{[3]} = \sigma(A3) = \sigma\Big(W_{(6, 12)}^{[3]} \times H_{(12, 1080)}^{[2]} + b_{(6, 1)}^{[3]}\Big)`

-  :math:`Output = Y_{(6, 1080)}`

.. code:: python

   def initialize_parameters():
       tf.set_random.seed(1)
       W1 = tf.get_variable(name = "W1", shape = [25, 12288], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = 1))
       b1 = tf.get_variable(name = "b1", shape = [25, 1], 
                            initializer = tf.zeros_initializer())
       W2 = tf.get_variable(name = "W2", shape = [12, 25], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = 1))
       b2 = tf.get_variable(name = "b2", shape = [12, 1], 
                            initializer = tf.zeros_initializer())
       W3 = tf.get_variable(name = "W3", shape = [6, 12], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = 1))
       b3 = tf.get_variable(name = "b3", shape = [6,1], 
                            initializer = tf.zeros_initializer())
       parameters = {
           "W1": W1,
           "b1": b1,
           "W2": W2,
           "b2": b2,
           "W3": W3,
           "b3": b3
       }
       return parameters

.. _header-n62:

3.循环执行
----------

前向传播：

-  :math:`Input = X_{(12288, 1080)}`

-  :math:`H_{(25, 1080)}^{[1]} = relu\Big(W_{(25, 12288)}^{[1]} \times X_{(12288, 1080)} + b_{(25, 1)}^{[1]}\Big)`

-  :math:`H_{(12, 1080)}^{[2]} = relu\Big(W_{(12, 25)}^{[2]} \times H_{(25, 1080)}^{[1]} + b_{(12, 1)}^{[2]}\Big)`

-  :math:`H_{(6, 1080)}^{[3]} = \sigma(A3) = \sigma\Big(W_{(6, 12)}^{[3]} \times H_{(12, 1080)}^{[2]} + b_{(6, 1)}^{[3]}\Big)`

-  :math:`Output = Y_{(6, 1080)}`

.. code:: python

   def forward_propagation(X, parameters):
       """
       Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
       """
       W1 = parameters["W1"]
       b1 = parameters["b1"]
       W2 = parameters["W2"]
       b2 = parameters["b2"]
       W3 = parameters["W3"]
       b3 = parameters["b3"]
       Z1 = tf.add(tf.matmul(W1, X), b1)
       A1 = tf.nn.relu(Z1)
       Z2 = tf.add(tf.matmul(W2, A1), b2)
       A2 = tf.nn.relu(Z2)
       Z3 = tf.add(tf.matmul(W3, A2), b3)
       return Z3

计算损失：

.. code:: python

   def compute_cost(Z3, Y):
       logits = tf.transpose(Z3)
       lables = tf.transpose(Y)
       cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
       return cost

反向传播、权重更新：

-  mini-batch

-  梯度下降算法、更新参数

.. code:: python

   def random_mini_batch(x, y, batch_size = 64, seed = 0):
       np.random.seed(seed)
       m = x.shape[1]
       # setp 1: shuffle (x, y)
       mini_batches = []
       permutation = list(np.random.permutation(m))
       shuffled_x = x[:, permutation]
       shuffled_y = y[:, permutation].reshape((1, m))
       # step 2: partition (shuffled_x, shuffled_y)
       num_complete_minibatches = math.floor(m / batch_size)
       for k in range(0, num_complete_minibatches):
           mini_batch_x = shuffled_x[:, 0:batch_size]
           mini_batch_y = shuffled_y[:, 0:batch_size]
           mini_batch = (mini_batch_x, mini_batch_y)
           mini_batches.append(mini_batch)
       if m % batch_size != 0:
           mini_batch_x = shuffled_x[:, 0:m - batch_size * math.floor(m / batch_size)]
           mini_batch_y = shuffled_y[:, 0:m - batch_size * math.floor(m / batch_size)]
           mini_batch = (mini_batch_x, mini_batch_y)
           mini_batches.append(mini_batch)

       return mini_batches


   def model(x_train, y_train, x_test, y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
       ops.reset_default_graph()

       # configuration
       tf.set_random_seed(1)
       seed = 3
       (n_x, m) = x_train.shape
       n_y = y_train.shape[0]
       costs = []

       X, Y = create_placeholders(n_x, n_y)
       # 初始化模型参数
       parameters = initialize_parameters()
       # 前向传播
       Z3 = forward_propagation(X, parameters)
       # 计算损失
       cost = compute_cost(Z3, Y)
       # 后向传播
       optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

       init = tf.global_variables_initializer()
       with tf.Session() as sess:
           sess.run(init)
           for epoch in range(num_epochs): # epoch: 0, 1, 2, ..., 1500
               epoch_cost = 0.0
               num_minibatches = int(m / minibatch_size)
               seed += 1
               minibatches = random_mini_batch(x_train, y_train, minibatch_size, seed)
               for minibatch in minibatches:
                   (minibatch_x, minibatch_y) = minibatch 
                   _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_x, Y: minibatch_y})
                   epoch_cost += minibatch_cost / num_minibatches

           if print_cost == True and epoch % 100 == 0:
               print("Cost after epoch %i: %f" % (epoch, epoch_cost))
           if print_cost == True and epoch % 5 == 0:
               costs.append(epoch_cost)

           # 模型结果可视化
           plt.plot(np.squeeze(costs))
           plt.ylabel('cost')
           plt.xlabel('iterations (per tens)')
           plt.title("Learning rate =" + str(learning_rate))
           plt.show()

           parameters = sess.run(parameters)        
           print ("Parameters have been trained!")

           correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
           accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))       
           print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))        
           print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))   

           return parameters

执行模型：

.. code:: python

   parameters = model(x_train, y_train, x_test, y_test)
