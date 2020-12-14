
NLP--词嵌入(word embedding)
======================================

基于神经网络的文本数据表示一般称为 **词向量(word vector)**、**词嵌入(word embedding)**、**分布式表示(distributed representation)**.

神经网络词向量与其他分布式类似，都基于分布式表达方式，核心依然是上下文的表示以及上下文与目标词之间的关系映射，
主要通过神经网络对上下文，以及上下文和目标词汇之间的关系进行建模。通过这种方式表示主要是由于神经网络的空间非常大，
所以这种方法可以表示复杂的上下文关系。

基于矩阵的表示方法，是较为常见的方法，但是无法表示出上下文之间的关联关系，所以随着词汇数量的增大，
空间复杂度会呈指数性增长。

1.词向量
-------------------------

NLP 相关任务中最常见的第一步是创建一个 **词表库** 并把每个词顺序编号。

- **One-hot** 词向量表示
    
    - One-hot 方法把每个词顺序编号，每个词就是一个很长的向量，向量的维度等于词表的大小，只有对应位置上的数字为1，其他都为 0。

        - 在实际应用中，一般采用稀疏编码存储，主要采用词的编号，这种表示方法一个最大的问题就是无法捕捉词与词之间的相似度，也称为“词汇鸿沟”问题，所以：

            - One-hot 的第一个问题是：One-hot 的基本假设是词之间的语义和语法关系是相互独立的，仅仅从两个向量是无法看出两个词汇之间的关系的，这种独立性不适合词汇语义的运算；

            - One-hot 的第二个问题是：维度爆炸问题，随着词典规模的增大，句子构成的词袋模型的维度变得越来越大，矩阵也变得超稀疏，这种维度的爆增，会大大耗费计算资源。

- **分布式** 词向量表示

    - 词汇分布式表示最早由 Hinton 在 1986 年提出，其基本思想是：通过训练将每个词映射成 K 维实数向量(K 一般为模型中的超参数)，
      通过词之间的距离(如，consine 相似度、欧氏距离)来判断它们之间的语义相似度。其中，word2vec 使用的就是这种分布式表示的词向量表示方式。

2.word2vec
-------------------------

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


3.词向量模型
-------------------------


4.CBOW 和 Skip-gram 模型
-------------------------




5.训练词向量
-----------------------------------------------------

5.1 word2vec 版本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


5.2 Gensim word2vec 示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 使用中文维基百科语料库作为训练库



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

