
NLP--文本向量化
=====================================

1.概述
----------------------------------------------------------------

   文本表示是 NLP 中的基础工作，文本表示的好坏直接影响到整个 NLP 系统的性能。

   在 NLP 研究领域，文本向量化是文本表示的一种重要方式。顾名思义，文本向量化就是将文本表示成一系列能够表达文本语义的向量，
   无论是中文还是英文，词语都是表达文本处理的最基本单元。

   随着互联网技术的发展，互联网上调度数据急剧增加。大量无标记的数据产生，使得研究者将注意力转移到利用无标注数据挖掘有价值的信息上来。
   词向量(``word2vec``)技术就是利用神经网络从大量无标注的文本中提取有用信息而产生的。

   一般来说词语是表达语义的基本单元。因为词袋模型只是将词语符号化，所以词袋模型是不包含任何语义信息的。
   如何使“词表示”包含语义信息是该领域研究者面临的问题。

      - 分布式假说(distributional hypothesis) 的提出为解决上面的问题提供了理论基础。该假说的核心思想是：上下文相似的词，
        其语义也相似。随后有学者整理了利用上下文分布词表示词义的方法，这类方法就是有名的 **词空间模型(word space model)**
      - 随着各类硬件设备计算能力的提升和相关算法的发展，神经网络模型逐渐在各个领域中崭露头角，可以灵活地对上下文进行建模是神经网络构造词表示的最大优点
      - 通过语言模型构建上下文与目标词之间的关系是一种常见的方法。神经网络词向量模型就是根据上下文与目标词之间的关系进行建模

   文本向量化的方法有很多：

      - 基于统计的方法

         - One-Hot encoding
         - 词袋模型

      - 基于神经网络的方法

         - ``word2vec``
         - ``doc2vec``/``str2vec``
            - 也有相当一部分研究者将文章或者句子作为文本处理的基本单元，于是产生了 ``doc2vec`` 和 ``str2vec``

2.NLP 特征工程
----------------------------------------------------------------

2.0 NLP 数据预处理简介
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 处理内容

      - 处理语料库文章
      - 处理语料库句子

   - 处理工具

      - 正则表达式

         - re.match
         - re.search

   - 处理方法

      - 语义解析器
      - 词频-逆向文件频率 (Term Frequency-Inverse Document Frequency, TF-IDF)
      - 词向量 word to vector(word2vec, Google-Tomas Mikolov, 2013)
      - 词嵌入 word embedding

2.1 One-Hot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- One-Hoe 算法简介

   - 在一个语料库中，给每个字、词编码一个索引，根据索引进行 One-Hot 表示。

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
      
         # John likes to watch moives, Mary likes too.

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

   4. 文本 One-hot 的缺点

      - 当语料库非常大时，需要建立一个很大的字典对所有单词进行索引编码。
        比如 100W 个单词，每个单词就需要表示成 100W 维的向量，而且这
        个向量是很稀疏的，只有一个地方为 1 其他全为 0。还有很重要的一点，
        这种表示方法无法表达单词与单词之间的相似程度，如 beautiful 和 
        pretty 可以表达相似的意思但是 One-Hot 无法将之表示出来。​

- One-Hot 算法 Python 实现

   .. code-block:: python

      from sklearn import CountVectorizer


2.2 词袋模型(Bag of Word)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 词袋模型算法

   - 词袋(Bag Of Word) 模型是最早的以词语为基本单元的文本向量化方法。词袋模型，也称为计数向量表示(Count Vectors).
     文档的向量表示可以直接使用单词的向量进行求和得到。

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

   4. 上述向量与原来文本中单词出现的顺序没有关系，而是词典中每个单词在文本中出现的频率。该方法虽然简单易行，
      但是存在如下三方面的问题：

      - 维度灾难
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


2.3 Bi-gram、 N-gram
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

   - 词袋模型、Bi-gram、N-gram 都是基于计数得到的，而 TF-IDF 则是基于频率统计得到的
   - TF-IDF 的分数代表了词语在当前文档和整个语料库中的相对





- TF-IDF 算法 Python 实现

   .. code-block:: python

      from sklearn import TfidfVectorizer

      # word level tf-idf
      tfidf_vect = TfidfVectorizer(analyzer = "word", token_pattern = r"\w{1,}", max_features = 5000)


      # n-gram level tf-idf



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


2.6 分布式表示
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




2.7 NNLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   通过语言模型构建上下文与目标词之间的关系是一种常见的方法。神经网络词向量模型就是根据上下文与目标词之间的关系进行建模。

   神经网路语言模型(Neural Network Language Model, NNLM) 是在研究者使用神经网络求解二元语言模型时提出来的。与传统方法估算下述概率不同:
   
   .. math:: 
   
      P(\omega_{i}|\omega_{i-(n-1)}, \cdots, \omega_{i-1})
   
   NNLM 模型直接通过一个神经网络结构对 :math:`n` 元条件概率进行估计。NNLM 模型的基本结构如下：

   .. image:: ../../images/NNLM.png


2.8 CBOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2.9 层级 Softmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.10 负例采样(Negative Sampling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.11 Skip-gram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.12 Fasttext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.12.1 fasttext 算法简介
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

2.12.2 fasttext 的 Python 实现
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

2.13 word2vec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 词到向量

   从深度学习的角度看,假设将 NLP 的语言模型看作是一个监督学习问题：给定上下文词 :math:`X`,输出中间词 :math:`Y`；
   或者给定中间词 :math:`X`,输出上下文词 :math:`Y`.基于输入 :math:`X` 和输出 :math:`Y` 之间的映射便是语言模型.
   这样的一个语言模型的目的便是检查 :math:`X` 和 :math:`Y` 放在一起是否符合自然语言规则,更通俗一点就是 :math:`X` 和
   :math:`Y` 放在一起是不是人话.

   所以,基于监督学习的思想,word2vec 便是一种基于神经网络训练的自然语言模型.word2vec 是谷歌于 2013 年提出的一种 NLP
   工具,其特点就是将词汇进行向量化,这样就可以定量的分析和挖掘词汇之间的联系.因而 word2vec 也是词嵌入表征的一种,
   只不过这种向量表征需要经过神经网络训练得到.

   word2vec 训练神经网路得到的一个关于输入 :math:`X` 和输出 :math:`Y` 之间的语言模型,关注的重点并不是说要把这个模型训练的有多好,
   而是要获取训练好的神经网络权重,这个权重就是我们要拿来对输入词汇 :math:`X` 的向量化表示.一旦拿到了训练预料所有词汇的词向量,接下来开展
   NLP 分析工作就相对容易一些.


2.14 word embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   基于神经网络的文本数据表示一般称为 **词向量(word vector)**、**词嵌入(word embedding)**、**分布式表示(distributed representation)**.

   神经网络词向量与其他分布式类似，都基于分布式表达方式，核心依然是上下文的表示以及上下文与目标词之间的关系映射，
   主要通过神经网络对上下文，以及上下文和目标词汇之间的关系进行建模。通过这种方式表示主要是由于神经网络的空间非常大，
   所以这种方法可以表示复杂的上下文关系。

   基于矩阵的表示方法，是较为常见的方法，但是无法表示出上下文之间的关联关系，所以随着词汇数量的增大，
   空间复杂度会呈指数性增长。

2.14.1 词向量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   NLP 相关任务中最常见的第一步是创建一个 **词表库** 并把每个词顺序编号。

   - **One-hot** 词向量表示
      
      - One-hot 方法把每个词顺序编号，每个词就是一个很长的向量，向量的维度等于词表的大小，只有对应位置上的数字为1，其他都为 0。

         - 在实际应用中，一般采用稀疏编码存储，主要采用词的编号，这种表示方法一个最大的问题就是无法捕捉词与词之间的相似度，也称为“词汇鸿沟”问题，所以：

               - One-hot 的第一个问题是：One-hot 的基本假设是词之间的语义和语法关系是相互独立的，仅仅从两个向量是无法看出两个词汇之间的关系的，这种独立性不适合词汇语义的运算；

               - One-hot 的第二个问题是：维度爆炸问题，随着词典规模的增大，句子构成的词袋模型的维度变得越来越大，矩阵也变得超稀疏，这种维度的爆增，会大大耗费计算资源。

   - **分布式** 词向量表示

      - 词汇分布式表示最早由 Hinton 在 1986 年提出，其基本思想是：通过训练将每个词映射成 K 维实数向量(K 一般为模型中的超参数)，
         通过词之间的距离(如，consine 相似度、欧氏距离)来判断它们之间的语义相似度。其中，word2vec 使用的就是这种分布式表示的词向量表示方式。

2.14.2 word2vec
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


2.14.3 词向量模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.14.4 CBOW 和 Skip-gram 模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.14.5 训练词向量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

2.15 para2vec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 段落到向量

2.16 doc2vec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 文章到向量

2.17 GloVe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 通过余弦函数、欧几里得距离来获得相似词的库
