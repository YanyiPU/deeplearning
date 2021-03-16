
RNN-LSTM
=======================

1. LSTM
-----------------------

1.2 LSTM：让RNN具备更好的记忆机制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  梯度爆炸和梯度消失对 RNN 的影响非常大，当 RNN 加深时，因为梯度消失的问题使得前层的网络权重得不到更新，RNN 的记忆性就很难生效

   -  在传统的 RNN 基础上，研究人员给出了一些著名的改进方案，即 RNN 变种网络，比较著名的是 GRU(循环门控单元)和 LSTM(长短期记忆网络)；GRU 和 LSTM 二者的结构基本一致，但有些许不同

   -  LSTM

      -  LSTM 的本质是一种 RNN 网络

      -  LSTM 在传统的 RNN 结构上做了相对复杂的改进，这些改进使得 LSTM 相对于经典的 RNN 能够很好地解决
         梯度爆炸和梯度消失的问题，让 RNN 具备更好的记忆性能




2.2.1 LSTM 结构
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**经典RNN结构与LSTM结构对比：**

   相对经典RNN，LSTM单元中包含了4个交互的网络层；

**LSTM结构数学表示：**

   - forget-gate:

      .. math::
         
         \Gamma_{f}^{(t)} = \sigma_f (W_f [a^{(t-1)}, x^{(t)}] + b_{f})

   - update-gate:

      .. math::

         \Gamma_{u}^{(t)} = \sigma_u (W_u [a^{(t-1)}, x^{(t)}] + b_{u})

      .. math::

         \widetilde{c}^{(t)} = \tanh (W_c [a^{(t-1)}, x^{(t)}] + b_{c})

   - remember-cell update:

      .. math::

         c^{(t)} = \Gamma_{f}^{(t)} \circ c^{(t-1)} + \Gamma_{u}^{(t)} \circ \widetilde{c}^{(t)}

   - output-gate:

      .. math::
         
         \Gamma_{o}^{(t)} = \sigma_o (W_o [a^{(t-1)}, x^{(t)}] + b_{o})

      .. math::

         a^{(t)} = \Gamma_{o}^{(t)} \circ \tanh(c^{(t)})

   - Output:

      .. math::
         
         y^{(t)} = softmax(a^{(t)})

**LSTM结构分解：**

   -  记忆细胞(remember cell)

      -  在LSTM单元的最上层有一条贯穿的关于记忆细胞 :math:`c^{(t-1)}` 到 :math:`c^{(t)}` 的箭头直线；这样贯穿的直线表现记忆信息在网络各层之间保持下去很容易

      -  记忆细胞表示：

      .. math::
         
         c^{(t-1)} \rightarrow c^{(t)}

   -  遗忘门(forget gate)

      -  所谓的遗忘门就是要决定从记忆细胞中是否丢弃某些信息；通过一个sigmoid函数处理；

      -  遗忘门接受来自输入 :math:`x^{(t)}` 和上一层隐状态 :math:`a^{(t-1)}` 的值进行加权计算处理；

      -  遗忘门计算公式：

      .. math::
         
         \Gamma_{f}^{(t)} = \sigma_f (W_f [a^{(t-1)}, x^{(t)}] + b_{f})

   -  更新门(update gate)

      -  更新们就是需要确定什么样的信息能够存入细胞状态中，这跟GRU中类似，除了计算更新们之外，还需要通过\ :math:`tanh` 计算记忆细胞的候选值 :math:`\widetilde{c^{(t)}}`；

      -  更新门计算公式：

      .. math::
         
         \Gamma_{u}^{(t)} = \sigma_u (W_u [a^{(t-1)}, x^{(t)}] + b_{u})

      .. math::

         \widetilde{c}^{(t)} = \tanh (W_c [a^{(t-1)}, x^{(t)}] + b_{c})

   LSTM结合遗忘门、更新门、上一层的记忆细胞和记忆细胞候选值来共同决定和更新当前细胞状态：

   .. math::
      
      c^{(t)} = \Gamma_{f}^{(t)} \circ c^{(t-1)} + \Gamma_{u}^{(t)} \circ \widetilde{c}^{(t)}

   -  输出门(output)

      -  LSTM提供了单独的输出门；

      -  输出门计算公式：

      .. math::

         \Gamma_{o}^{(t)} = \sigma_o (W_o [a^{(t-1)}, x^{(t)}] + b_{o})

      .. math::

         a^{(t)} = \Gamma_{o}^{(t)} \circ \tanh(c^{(t)})
