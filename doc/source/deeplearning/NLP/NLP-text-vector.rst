
NLP--文本向量化
================================================================

1.概述
----------------------------------------------------------------

   - 文本向量化：

      - 文本向量化又称为 “词向量模型”、“向量空间模型”，即将文本表示成计算机可识别的实数向量，
        根据粒度大小不同，可将文本特征表示分为字、词、句子、篇章几个层次
      - 文本向量化方法一般称为词嵌入(word embedding)方法，词嵌入这个说法很形象，就是把文本中的词嵌入到文本空间中，用一个向量来表示词

   - 文本向量化的方法有很多：

      - 离散表示
      
         - 基于规则、统计

            - 词集模型(Set of Word)
               
               - One-Hot encoding
               - 统计各词在句子中是否出现

            - 词袋模型(Bag of Word)
               
               - 统计各词在句子中出现的次数

            - Bi-gram、N-gram

            - TF-IDF
               
               - 统计各词在文档中的 TF-IDF 值(词袋模型 + IDF 值)
            
            - 共现矩阵
            
      - 分布式表示

         - 基于神经网络

            - ``word2vec``
            - ``doc2vec``
            - ``str2vec``

   .. note:: 

      Tomas Mikolov 2013 年在 ICLR 提出用于获取 word vector 的论文《Efficient estimation of word representations in vector space》，
      文中简单介绍了两种训练模型 CBOW、Skip-gram，以及两种加速方法 Hierarchical Softmax、Negative Sampling。
      除了 word2vec 之外，还有其他的文本向量化的方法，因此在这里做个总结.


   NLP 相关任务中最常见的第一步是创建一个 **词表库** 并把每个词顺序编号。

   - **One-hot** 词向量表示
      
      - One-hot 方法把每个词顺序编号，每个词就是一个很长的向量，向量的维度等于词表的大小，只有对应位置上的数字为1，其他都为 0。

         - 在实际应用中，一般采用稀疏编码存储，主要采用词的编号，这种表示方法一个最大的问题就是无法捕捉词与词之间的相似度，也称为“词汇鸿沟”问题，所以：

               - One-hot 的第一个问题是：One-hot 的基本假设是词之间的语义和语法关系是相互独立的，仅仅从两个向量是无法看出两个词汇之间的关系的，这种独立性不适合词汇语义的运算；

               - One-hot 的第二个问题是：维度爆炸问题，随着词典规模的增大，句子构成的词袋模型的维度变得越来越大，矩阵也变得超稀疏，这种维度的爆增，会大大耗费计算资源。

   - **分布式** 词向量表示

      - 词汇分布式表示最早由 Hinton 在 1986 年提出，其基本思想是：通过训练将每个词映射成 K 维实数向量(K 一般为模型中的超参数)，
         通过词之间的距离(如，consine 相似度、欧氏距离)来判断它们之间的语义相似度。其中，word2vec 使用的就是这种分布式表示的词向量表示方式。



2.离散表示
-----------------------------------------

   文本向量化离散表示是一种基于规则和统计的向量化方式，常用的方法包括 **词集模型** 和 **词袋模型**，
   都是基于词之间保持独立性、没有关联为前提，将所有文本中单词形成一个字典，然后根据字典来统计单词出现频数，
   不同的是：

      - 词集模型：

         - 统计各词在句子中是否出现
         - 例如 One-Hot Representation，只要单个文本中单词出现在字典中，就将其置为 1，不管出现多少次
      
      - 词袋模型：
         
         - 统计各词在句子中出现的次数
         - 只要单个文本中单词出现在字典中，就将其向量值加 1，出现多少次就加多少次
   
   其基本的特点是忽略了文本信息中的语序信息和语境信息，仅将其反映为若干维度的独立概念，
   这种情况有着因为模型本身原因而无法解决的问题，比如主语和宾语的顺序问题，
   词袋模型天然无法理解诸如“我为你鼓掌”和“你为我鼓掌”两个语句之间的区别。

2.1 One-Hot Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- One-Hoe 算法简介

      - 在一个语料库中，给每个字、词编码一个索引，建立一个语料库字典，然后根据词在字典中的索引进行 One-Hot 表示
      - One-Hot 将每个字、词都表示成一个长向量，向量的维度是词典的大小，词的当前位置用 1 表示，其他位置用 0 表示
      - 如果只需要表示文本语料中的单词，可以只对其中出现过的单词进行索引编码即可

- One-Hot 算法示例

   1. 文本语料

      .. code-block:: python
      
         John likes to watch moives, Mary likes too.
         John also likes to watch football games.
   
   2. 基于上述两个文档中出现的单词，构建如下词典(dictionary)

      .. code-block:: python
         
         {
            "John": 1, 
            "likes": 2,
            "to": 3,
            "watch": 4,
            "moives": 5,
            "also": 6,
            "football": 7,
            "games": 8,
            "Mary": 9,
            "too": 10,
         }

   3. 文本 One-Hot

      .. code-block:: python
      
         # John likes to watch movies, Mary likes too.

         John:     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         likes:    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
         to:       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
         watch:    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
         movies:   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
         also:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         football: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         games:    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         Mary:     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
         too:      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

      .. code-block:: python
      
         # John also likes to watch football games.

         John:     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         likes:    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
         to:       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
         watch:    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
         movies:   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         also:     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
         football: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
         games:    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
         Mary:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         too:      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

   4. 文本 One-hot 的优、缺点

      - 优点：简单快捷
      - 缺点：数据稀疏、耗时耗空间、不能很好地展示词与词之间的相似关系，且还未考虑到词出现的频率，因而无法区别词的重要性

- One-Hot 算法 Python 实现

   .. code-block:: python

      import os
      import numpy as np
      import pandas as pd
      import jieba
      import config

      def doc2onthot_matrix(file_path):
         """
         文本向量化 One-Hot
            1.文本分词
         """
         # (1)读取待编码的文件
         with open(file_path, encoding = "utf-8") as f:
            docs = f.readlines()

         # (2)将文件每行分词，分词后的词语放入 words 中
         words = []
         for i in range(len(docs)):
            docs[i] = jieba.lcut(docs[i].strip("\n"))
            words += docs[i]
         
         # (3)找出分词后不重复的词语，作为词袋，是后续 onehot 编码的维度, 放入 vocab 中
         vocab = sorted(set(words), key = words.index)

         # (4)建立一个 M 行 V 列的全 0 矩阵，M 是文档样本数，这里是行数，V 为不重复词语数，即编码维度
         M = len(docs)
         V = len(vocab)
         onehot = np.zeros((M, V))
         for i, doc in enumerate(docs):
            for word in doc:
               if word in vocab:
                  pos = vocab.index(word)
                  onehot[i][pos] = 1
         onehot = pd.DataFrame(onehot, columns = vocab)
         return onehot


      if __name__ == "__main__":
         corpus = os.path.join(config.data_dir, "corpus.txt")
         onehot = doc2onthot_matrix(corpus)
         print(onehot)


   .. code-block:: python

      from sklearn import DictVectorizer


2.2 词袋模型(Bag of Word)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 词袋模型算法

   - 对于句子、篇章，常用的离散表示方法是词袋模型，词袋模型以 One-Hot 为基础，忽略词表中词的顺序和语法关系，
     通过记录词表中的每一个词在该文本中出现的频次来表示该词在文本中的重要程度，解决了 One-Hot 未能考虑词频的问题
   - 词袋(Bag Of Word) 模型是最早的以词语为基本单元的文本向量化方法。词袋模型，也称为计数向量表示(Count Vectors).
     文档的向量表示可以直接使用单词的向量进行求和得到

- 词袋模型示例

   1. 文本语料

      .. code-block:: python
        
         John likes to watch movies, Mary likes too.
         John also likes to watch football games.

   2. 基于上述两个文档中出现的单词，构建如下词典(dictionary)

      .. code-block:: python

         {
            "John": 1, 
            "likes": 2,
            "to": 3,
            "watch": 4,
            "movies": 5,
            "also": 6,
            "football": 7,
            "games": 8,
            "Mary": 9,
            "too": 10,
         }

   3. 上面词典中包含 10 个单词，每个单词有唯一的索引，那么每个文本可以使用一个 10 维的向量来表示:

      .. code-block:: python

         John likes to watch movies, Mary likes too.  ->  [1, 2, 1, 1, 1, 0, 0, 0, 1, 1]
         John also likes to watch football games.     ->  [1, 1, 1, 1, 0, 1, 1, 1, 0, 0]

      ============================================= ====== ====== === ====== ======= ===== ========= ====== ===== ====
       文本                                          John   likes  to  watch  movies  also  football  games  Mary  too
      ============================================= ====== ====== === ====== ======= ===== ========= ====== ===== ====
       John likes to watch movies, Mary likes too.  [1,    2,     1,  1,     1,      0,    0,        0,     1,    1]
       John also likes to watch football games.     [1,    1,     1,  1,     0,      1,    1,        1,     0,    0]
      ============================================= ====== ====== === ====== ======= ===== ========= ====== ===== ====

      - 横向来看，把每条文本表示成了一个向量
      - 纵向来看，不同文档中单词的个数又可以构成某个单词的词向量, 如: "John" 纵向表示成 ``[1, 1]``

   4. 词袋模型优缺点

      - 优点：

         - 方法简单，当语料充足时，处理简单的问题如文本分类，其效果比较好

      - 缺点：

         - 数据稀疏、维度大
         - 无法保留词序信息
         - 存在语义鸿沟的问题

- 词袋模型 Python 实现

   .. code-block:: python

      from sklearn import CountVectorizer
      
      count_vect = CountVectorizer(analyzer = "word")
      
      # 假定已经读进来 DataFrame，"text"列为文本列
      count_vect.fit(trainDF["text"])

      # 每行为一条文本，此句代码基于所有语料库生成单词的词典
      xtrain_count = count_vect.transform(train_x)


2.3 Bi-gram、N-gram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Bi-gram、N-gram 算法简介

   - 与词袋模型原理类似，Bi-gram 将相邻两个词编上索引，N-gram 将相邻 N 个词编上索引

- Bi-gram、N-gram 算法示例

   1. 文本语料

      .. code-block:: python
        
         John likes to watch movies, Mary likes too.
         John also likes to watch football games.

   2. 基于上述两个文档中出现的单词，构建如下词典(dictionary)

      .. code-block:: python

         {
            "John likes": 1,
            "likes to": 2,
            "to watch": 3,
            "watch movies": 4,
            "Mary likes": 5,
            "likes too": 6,
            "John also": 7,
            "also likes": 8,
            "watch football": 9,
            "football games": 10,
         }

   3. 上面词典中包含 10 组单词，每组单词有唯一的索引，那么每个文本可以使用一个 10 维的向量来表示:

      .. code-block:: python

         John likes to watch movies. Mary likes too.  -> [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
         John also likes to watch football games.     -> [0, 1, 1, 0, 0, 0, 1, 1, 1, 1]

   4. Bi-gram、N-gram 优点

      - 考虑了词的顺序

   5. Bi-gram、N-gram 缺点

      - 词向量急剧膨胀

- Bi-gram、N-gram 算法 Python 实现

   .. code-block:: python

      from . import .

2.4 TF-IDF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- TF-IDF 算法简介

   - TF-IDF(词频-逆文档频率法，Term Frequency-Inverse Document Frequency) 作为一种加权方法，
     TF-IDF 在词袋模型的基础上对次出现的频次赋予 TF-IDF 权值，对词袋模型进行修正，进而表示该词在文档集合中的重要程度
      
      - 统计各词在文档中的 TF-IDF 值(词袋模型 + IDF 值)
      - 词袋模型、Bi-gram、N-gram 都是基于计数得到的，而 TF-IDF 则是基于频率统计得到的
      - 在利用 TF-IDF 进行特征提取时，若词 α 在某篇文档中出现频率较高且在其他文档中出现频率较低时，
        则认为α可以代表该文档的特征，具有较好的分类能力，那么α作为特征被提取出来
   
   - TF-IDF 的分数代表了词语在当前文档和整个语料库中的相对重要性。TF-IDF 分数由两部分组成

      - TF(Term Frequency)：词语频率

         .. math::

            TF(t) = \frac{词语在当前文档出现的次数}{当前文档中词语的总数}
         
         - TF 判断的是该字/词语是否是当前文档的重要词语，但是如果只用词语出现频率来判断其是否重要可能会出现一个问题，
           就是有些通用词可能也会出现很多次，如：a、the、at、in 等，当然一般我们会对文本进行预处理时去掉这些所谓的停用词(stopwords)，
           但是仍然会有很多通用词无法避免地出现在很多文档中，而其实它们不是那么重要

      - IDF(Inverse Document Frequency)：逆文档频率

         .. math::

            IDF(t) = log_{e} \Big(\frac{文档总数}{出现该词语的文档总数} \Big)

         - IDF 用于判断是否在很多文档中都出现了词词语，即很多文档或所有文档中都出现的就是通用词。
           出现该词语的文档越多，IDF 越小，其作用是抑制通用词的重要性

      - 将上述求出的 TF 和 IDF 相乘得到的分数 TF-IDF，就是词语在当前文档和整个语料库中的相对重要性
      - TF-IDF 与一个词在当前文档中出现次数成正比，与该词在整个语料库中的出现次数成反比

- TF-IDF 算法优缺点

   - 优点：
      
      - 简单快速，结果比较符合实际情况
   
   - 缺点：
      
      - 单纯以"词频"衡量一个词的重要性，不够全面，有时重要的词可能出现次数并不多
      - 无法体现词的位置信息，出现位置靠前的词与出现位置靠后的词，都被视为重要性相同，这是不正确的

- TF-IDF 算法 Python 实现

   .. code-block:: python

      from sklearn import TfidfVectorizer
      from sklearn import HashingVectorizer

      # word level tf-idf
      tfidf_vect = TfidfVectorizer(analyzer = "word", token_pattern = r"\w{1,}", max_features = 5000)
      tfidf_vect.fit(trianDF["text"])
      xtrain_tfidf = tfidf_vect.transform(train_x)

      # n-gram level tf-idf
      tfidf_vect_ngram = TfidfVectorizer(analyzer = "word", token_pattern = r"\w{1,}", ngram_ragne(2, 3), max_features = 5000)
      tfidf_vect_ngram.fit(trainDF["text"])
      xtrain_tfidf = tfidf_vect.transform(train_x)



2.5 共现矩阵(Co-currence Matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 共现矩阵算法简介

   - 共现：即共同实现，比如：一句话中共同出现，或一篇文章中共同出现
   - 共现矩阵构造时需要给出共同出现的距离一个规范-- **窗口**

      - 如果窗口宽度是 2，那就是在当前词的前后各 2 个词的范围内共同出现，
        可以想象，其实是一个总长为 5 的窗口依次扫过所有文本，同时出现在其中的词就说它们共现

   - 当前词与自身不存在共现，共现矩阵实际上是对角矩阵
      
      - 实际应用中，用共现矩阵的一行(列)作为某个词的词向量，其向量维度还是会随着字典大小呈线性增长，而且存储共现矩阵可能需要消耗巨大的内存
      - 一般配合 PCA 或者 SVD 将其进行降维，比如：将 :math:`m \times n` 的矩阵降维为 :math:`m \times r`，其中 :math:`r \le n`，即将词向量的长度进行缩减

         .. image:: ../../images/SVG.png

- 共现矩阵算法示例

   1. 文本语料

      .. code-block:: python

         John likes to watch movies.
         John likes to play basketball.

   2. 假设上面两句话设置窗口宽度为 1，则共现矩阵如下

      ============= ===== ====== === ====== ======= ===== ===========
       共现矩阵      John  likes  to  watch  moives  play  basketball
      ============= ===== ====== === ====== ======= ===== ===========
       John         0     2      0   0      0       0     0
       likes        2     0      2   0      0       0     0
       to           0     2      0   1      0       1     0
       watch        0     0      1   0      1       0     0
       moives       0     0      0   1      0       0     0
       play         0     0      1   0      0       0     1
       basketball   0     0      0   0      0       1     0
      ============= ===== ====== === ====== ======= ===== ===========

- 共现矩阵算法 Python 实现

   .. code-block:: python

      from . import .


3.分布式表示
-----------------------------------------

   离散表示虽然能够进行词语或者文本的向量表示，进而用模型进行情感分析或者是文本分类之类的任务。
   但其不能表示词语间的相似程度或者词语间的类比关系。
   
      - 比如：beautifule 和 pretty 两个词语，它们表达相近的意思，所以希望它们在真个文本的表示空间内挨得很近。
   
   一般认为，词向量、文本向量之间的夹角越小，两个词相似度越高，词向量、文本向量之间夹角的关系用下面的余弦夹角进行表示：

      .. math::

         \cos \theta = \frac{\overrightarrow{A} \cdot \overrightarrow{B}}{|\overrightarrow{A}| \cdot |\overrightarrow{B}|}

   离散表示，如 One-Hot 表示无法表示上面的余弦关系，引入分布式表示方法，其主要思想是 **用周围的词表示该词**.

3.1 NNLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


   2003 年提出了神经网路语言模型(Neural Network Language Model, NNLM)，
   其用前 :math:`n-1` 个词预测第 :math:`n` 个词的概率，并用神经网络搭建模型。

   - 目标函数
   
      .. math:: 
      
         L(\theta) = \sum_{t} log P(\omega_{t}|\omega_{t-n}, \omega_{t-n+1}, \cdots, \omega_{t-1})
   
      - 使用非对称的前向窗口，长度为 :math:`n-1` ，滑动窗口遍历整个语料库求和，使得目标概率最大化，其计算量正比于语料库的大小。
        同时，预测所有词的概率综合应为 1。

      .. math::

         \sum_{\omega \in \{vocabulary\}} P(\omega|\omega_{t-n+1}, \cdots, \omega_{t-1})

   - NNLM 模型的基本结构如下：

      .. image:: ../../images/NNLM.png

   - 样本的一组输入是第 :math:`n` 个词的前 :math:`n-1` 个词的 One-Hot表示，目标是预测第 :math:`n`  个词，
     输出层的大小是语料库中所有词的数量，然后 sotfmax 回归，使用反向传播不断修正神经网络的权重来最大化第 :math:`n`  个词的概率。
     当神经网络学得到权重能够很好地预测第 :math:`n` 个词的时候，输入层到映射层，即这层，其中的权重 Matrix C 被称为投影矩阵，
     输入层各个词的 Ont-Hot 表示法只在其对应的索引位置为 1，其他全为 0，在与 Matrix C 矩阵相乘时相当于在对应列取出列向量投影到映射层。

      .. math::

         Matrix C = (w_{1}, w_{2}, \cdots, w_{v}) = 
         \begin{bmatrix}
         (\omega_{1})_{1} & (\omega_{2})_{1} & \cdots & (\omega_{v})_{1} \\
         (\omega_{1})_{2} & (\omega_{2})_{2} & \cdots & (\omega_{v})_{2} \\
         \vdots           & \vdots           &        & \vdots           \\
         (\omega_{1})_{D} & (\omega_{2})_{D} & \cdots & (\omega_{v})_{D} \\
         \end{bmatrix}

   此时的向量​就是原词​的分布式表示，其是稠密向量而非原来 One-Hot 的稀疏向量了。

   在后面的隐藏层将这 n-1 个稠密的词向量进行拼接，如果每个词向量的维度为 D，则隐藏层的神经元个数为 (n-1)×D，
   然后接一个所有待预测词数量的全连接层，最后用 softmax 进行预测。

   可以看到，在隐藏层和分类层中间的计算量应该是很大的，word2vec 算法从这个角度出发对模型进行了简化。
   word2vec 不是单一的算法，而是两种算法的结合：连续词袋模型（CBOW）和跳字模型（Skip-gram）。

3.2 CBOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CBOW 和 Skip-gram 模型的原理示意图：

   .. image:: ../../images/CBOW_Skip_gram.png

连续词袋模型在 NNLM 基础上有以下几点创新：

   1.取消了隐藏层，减少了计算量
   2.采用上下文划窗而不是前文划窗，即用上下文的词来预测当前词
   3.投影层不再使用各向量拼接的方式，而是简单的求和平均

其目标函数为：

   .. math::

      J = \sum_{\omega \in corpus} P(\omega | context(\omega))

可以看到，上面提到的取消隐藏层，投影层求和平均都可以一定程度上减少计算量，但输出层的数量在那里，
比如语料库有 500W 个词，那么隐藏层就要对 500W 个神经元进行全连接计算，这依然需要庞大的计算量。
word2vec 算法又在这里进行了训练优化。

   .. image:: ../../images/CBOW.png

3.3 层级 Softmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 霍夫曼树

   霍夫曼树是一棵特殊的二叉树，了解霍夫曼树之前先给出几个定义：

      - 路径长度：在二叉树路径上的分支数目，其等于路径上结点数-1
      - 结点的权：给树的每个结点赋予一个非负的值
      - 结点的带权路径长度：根结点到该结点之间的路径长度与该节点权的乘积
      - 树的带权路径长度：所有叶子节点的带权路径长度之和

   霍夫曼树的定义为：

      - 在权为 :math:`\omega_{1}, \omega_{2}, \cdots, \omega_{n}`  ​的​ :math:`n` 个叶子结点所构成的所有二叉树中，
        带权路径长度最小的二叉树称为最优二叉树或霍夫曼树

   可以看出，结点的权越小，其离树的根结点越远。

- 层级 Softmax

   word2vec 算法利用霍夫曼树，将平铺型 softmax 压缩成层级 softmax，不再使用全连接。
   具体做法是根据文本的词频统计，将词频赋给结点的权。

   在霍夫曼树中，叶子结点是待预测的所有词，在每个子结点处，用 sigmoid 激活后得到往左走的概率 p，
   往右走的概率为 1-p。最终训练的目标是最大化叶子结点处预测词的概率。

   层级 softmax 的实现有点复杂，暂时先搞清楚大致原理~

3.4 负例采样(Negative Sampling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   负例采样的想法比较简单，假如有 :math:`m` 个待预测的词，每次预测的一个正样本词，其他的 :math:`m-1` 个词均为负样本。
      
      - 一方面正负样本数差别太大
      - 另一方面，负样本中可能有很多不常用，或者词预测时概率基本为0的样本，我们不想在计算它们的概率上面消耗资源

   比如现在待预测的词有 100W 个，正常情况下，我们分类的全连接层需要 100W 个神经元，我们可以根据词语的出现频率进行负例采样，
   一个正样本加上采样出的比如说 999 个负样本，组成 1000 个新的分类全连接层。

   采样尽量保持了跟原来一样的分布，具体做法是将 :math:`[0, 1]` 区间均分为 108 份，然后根据词出现在语料库中的次数赋予每个词不同的份额。

      .. math:: 

         len(\omega) = \frac{counter(\omega)}{\sum_{\mu \in D} counter(\mu)}

   然后在 :math:`[0, 1]` 区间掷筛子，落在哪个区间就采样哪个样本。实际上，最终效果证明上式中取 :math:`counter(\omega)` 的 :math:`3/4` 次方效果最好，
   所以在应用汇总也是这么做的。

3.5 Skip-gram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Skip-gram 模型与 CBOW 模型相反，它是用当前词来预测上下文的词。也是用当前词的 One-Hot 向量经过投影矩阵得到其稠密表示，
   然后预测其周围词的 One-Hot 向量，即网络的输出有多个，就是投影层到分类层的权重，具体实践时，搭建好一个输出的网络之后，
   如果要预测周围的 4 个词，可以将 One-Hot 表示分别作为输出，进行 4 次网络的正向传播，然后使其 Loss 之和最小。

      .. image:: ../../images/skip_gram.png

3.6 Fasttext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.6.1 fasttext 算法简介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   fasttext 的模型与 CBOW 类似，实际上，fasttext 的确是由 CBOW 演变而来的。CBOW 预测上下文的中间词，fasttext 预测文本标签。
   与 word2vec 算法的衍生物相同，稠密词向量也是训练神经网路的过程中得到的。

      .. image:: ../../images/fasttext.png

   fasttext 的输入是一段词的序列，即一篇文章或一句话，输出是这段词序列属于某个类别的概率，所以，fasttext 是用来做文本分类任务的。

   fasttext 中采用层级 softmax 做分类，这与 CBOW 相同。fasttext 算法中还考虑了词的顺序问题，即采用 N-gram，
   与之前介绍的离散表示法相同，如：

      - 今天天气非常不错，Bi-gram 的表示就是：今天、天天、天气、气非、非常、常不、不错

   fasttext 做文本分类对文本的存储方式有要求：

      .. code-block:: 

         __label__1, It is a nice day.
         __label__2, I am fine, thank you.
         __label__3, I like play football.

      其中：

         - ``__label__``：为实际类别的前缀，也可以自己定义

3.6.2 fasttext 的 Python 实现
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - GitHub:
      
      - https://github.com/facebookresearch/fastText
   
   - 示例:

      .. code-block:: python

         classifier = fasttext.supervised(input_file, output, label_prefix = "__label__")
         result = classifier.test(test_file)
         print(result.precision, result.recall)


      其中：

         - ``input_file``：是已经按照上面的格式要求做好的训练集 txt
         - ``output``：后缀为 ``.model``，是保存的二进制文件
         - ``label_prefix``：可以自定类别前缀


3.7 Word Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   1. 什么是词嵌入
   2. 不同类型的词嵌入

      - 基于频率的词嵌入

         - Count Vector
         - TF-IDF
         - Co-Occurrence Matrix

      - 基于预测的词嵌入

         - CBOW
         - Skip-Gram
   
   3. 词嵌入使用示例
   4. 使用预训练的词向量
   5. 训练自己的词向量

3.7.1 词嵌入简介
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - Word Embedding 就是使用一个字典将一个词映射到一个向量

      - 句子

         - ``"Word Embedding are Word converted into numbers"``

      - 词:

         - ``"Embedding"``
         - ``"numbers"``

      - 字典: 句子中不重复词的列表

         - ``["Word", "Embedding", "are", "converted", "into", "numbers"]``

      - 向量

         - one-hot

            - ``"number"``: [0, 0, 0, 0, 0, 1]
            - ``"converted"``: [0, 0, 0, 1, 0, 0]


3.7.2 词嵌入类型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - 基于频率的词嵌入

      - Count Vector
      - TF-IDF Vector
      - Co-Occurrence Matrix

   - 基于预测的词嵌入

      - CBOW
      - Skip-Gram

3.7.3 基于频率的词嵌入
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.7.3.1 Count Vector
''''''''''''''''''''''''''''''''''''''''''

- 假设

   - 语料 C，包含 D 个文档: :math:`\{d_{1}, d_{2}, \cdots, d_{D}\}` 
   - 语料 C 中的 N 个不重复词构成字典
   - Count Vector matrix(计算向量矩阵) :math:`M_{D \times N}`
   - 计数向量矩阵 M 的第 :math:`i, i=1, 2, \cdots, D` 行包含了字典中每个词在文档 :math:`d_{i}` 中的频率

- 示例

   - 语料 (D = 2):

      - :math:`d_1` : He is a lazy boy. She is also lazy.
      - :math:`d_2` : Neeraj is a lazy person.

   - 字典 (N = 6):

      - ["He", "She", "lazy", "boy", "Neeraj", "person"]

   - 计数向量矩阵:

      ============== ===== ====== ====== ===== ========= =======
       CountVector   He    She    lazy   boy   Neeraj    person
      ============== ===== ====== ====== ===== ========= =======
      :math:`d_1`    1     1      2      1     0         0
      :math:`d_2`    0     0      1      0     1         1
      ============== ===== ====== ====== ===== ========= =======


3.7.5 word2vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``word2vec`` 简介

      - ``word2vec`` 是 Google 在 2013 年发布的一个开源词向量建模工具
      - ``word2vec`` 使用的算法是 Bengio 等人在 2001 年提出的 Neural Network Language Model(NNLM) 算法
      - ``word2vec`` 是一款将词表征为实数值向量的高效工具

- ``word2vec`` 核心思想

   - ``word2vec`` 以及其他词向量模型，都基于同样的假设：
      
      - (1) 衡量词语之间的相似性，在于相邻词汇是否相识，这是基于语言学的“距离象似性”原理。
      - (2) 词汇和它的上下文构成了一个象，当从语料库当中学习得到相识或者相近的象时，它们在语义上总是相识的。

   - ``word2vec`` 模型

      - CBOW(Continuous Bag-Of-Words, 连续的词袋模型)
      - Skip-Gram

- ``word2vec`` 优点

   - 高效，Mikolov 在论文中指出一个优化的单机版本一天可以训练上千亿个词


- 训练 ``word2vec``

   1.word2vec 版本

      - Google ``word2vec``

         - https://github.com/dav/word2vec

      - Gensim Python ``word2vec``

         - https://pypi.python.org/pypi/gensim

      - C++ 11

         - https://github.com/jdeng/word2vec

      - Java 

         - https://github.com/NLPchina/Word2VEC_java

   .. note:: 

      - ``word2vec`` 一般需要大规模语料库(GB 级别)，这些语料库需要进行一定的预处理，变为精准的分词，才能提升训练效果：

      - 常用大规模中文语料库：

         - 维基百科中文语料(5.7G xml) https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

               - 标题
               - 分类
               - 正文

         - 搜狗实验室的搜狗 SouGouT(5TB 网页原版) https://www.sogou.com/labs/resource/t.php


   2.Gensim word2vec 示例

      使用中文维基百科语料库作为训练库

         1. 数据预处理

            - 大概等待 15min 左右，得到 280819 行文本，每行对应一个网页

            .. code-block:: python

               from gensim.corpora import WikiCorpus

               space = " "
               with open("wiki-zh-article.txt", "w", encoding = "utf8") as f:
                     wiki = WikiCorpus("zhwiki-latest-pages-articles.xml.bz2", lemmatize = False, dictionary = {})
                     for text in wiki.get_texts():
                        f.write(space.join(text) + "\n")
               print("Finished Saved.")

         2. 繁体字处理

            - 目的：
               
               - 因为维基语料库里面包含了繁体字和简体字，为了不影响后续分词，所以统一转化为简体字
            
            - 工具
               
               - opencc(https://github.com/BYVoid/OpenCC)

         .. code-block:: shell

            opencc -i corpus.txt -o wiki-corpus.txt -c t2s.json


         3. 分词

            - jieba
            - ICTCLAS(中科院)
            - FudanNLP(复旦)

3.7.6 para2vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - 段落到向量

3.7.7 doc2vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - 文章到向量

3.8 GloVe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 通过余弦函数、欧几里得距离来获得相似词的库

4.参考文献
-----------------------------------------

   1. `Efficient Estimation of Word Representations in Vector Space <https://arxiv.org/pdf/1301.3781.pdf>`_ 
   2. `Bag of Tricks for Efficient Text Classification <https://arxiv.org/pdf/1607.01759.pdf>`_ 
   3. `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/pdf/1810.04805.pdf>`_ 
   4. `A Neural Probabilistic Language Model <https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf>`_ 
   5. `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ 