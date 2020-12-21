
NLP--Gensim
====================

1.Gensim 简介
-----------------------------------------

1.1 Gensim 简介
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Gensim is a Free python library

- Gensim is a topic modelling for humans

    - Train large-scale semantic NLP models

    - Represent text as semantic vectors

    - Find semantically related documents

- Why Gensim?

    - super fast

    - data streaming

    - platform independent

    - proven

    - open source

    - ready-to-use models and corpora

1.2 Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: python
    
        from gensim import corpora, models, similarities, downloader

        # Stream a training corpus directly from S3.
        corpus = corpora.MmCorpus("s3://path/to/corpus")

        # Train Latent Semantic Indexing with 200D vectors.
        lsi = models.LsiModel(corpus, num_topics = 200)

        # Convert another corpus to the LSI space and index it.
        index = similarities.MatrixSimilarity(lsi[another_corpus])

        # Compute similarity of a query vs indexed documents.
        sims = index[query]


2.Gensim 安装
-----------------------------------------

2.1 Gensim 安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. code-block:: shell

        $ pip install --upgrade gensim

        $ conda install -c conda-forge gensim

2.2 Gensim 依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Python 3.6, 3.7, 3.8

    - Numpy

    - smart_open: 用于打开远程存储库中的文件或压缩文件


3.Gensim 使用
-----------------------------------------

    .. code-block:: python

        import  pprint

3.1 Gensim 中常用核心概念
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gensim`` 的核心概念:

    - Document

    - Corpus

    - Vector

    - Model

3.1.1 Document
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code-block:: python

        document = "Human machine interface for lab abc computer applications"


3.1.2 Corpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一个 ``Corpus`` 是一系列 ``Document`` 对象的集合，Corpora 在 Gensim 中提供了两个角色：

    - 1.``Model`` 训练的输入。在训练期间，模型会使用该训练语料库来查找常见的主题和主题，从而初始化其内部模型参数

        - Gensim 专注于无监督模型，因此不需要人工干预，例如昂贵的注释或手工标记文档.

    - 2.整理好的 ``Document``。训练后，可以使用主题模型从新文档(训练语料库中未显示的文档)中提取主题

        - 可以为此类语料库索引 **相似性查询**，通过语义相似性查询，聚类等

    .. code-block:: python

        from gensim import corpora, models, similarities, downloader
        from collections import defaultdict

        # 1.语料库
        text_corpus = [
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time",
            "The EPS user interface management system",
            "System and human system engineering testing of EPS",
            "Relation of user perceived response time to error measurement",
            "The generation of random binary unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors IV Widths of trees and well quasi ordering",
            "Graph minors A survey",
        ]

        # 2.创建常用词集合
        # 停用词
        stoplist = set("for a of the and to in".spilt(" "))
        # 常用词
        texts = [[word for word in document.lower().split() if word not in stoplist] for document in text_corpus]

        # 3.统计词频
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        
        # 4.删除只出现一次的词
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
        pprint.pprint(processed_corpus)

        # 5.将语料库中的每个单词与唯一的整数ID相关联
        dictionary = corpora.Dictionary(processed_corpus)
        print(dictionary)


3.1.3 Vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了推断语料库中的潜在结构，需要可以表示文档的数学处理方式：

    - 方法 1: 将文档表示为 **特征向量**

        - 密集向量

    - 方法 2: 词袋模型

        - 稀疏向量/词袋向量



    .. code-block:: python
    
        pprint.pprint(dictionary.token2id)


使用 ``doc2bow`` 为文档创建单词袋表示法:


    .. code-block:: python
    
        new_doc = "Human computer interaction"
        new_vec = dictionary.doc2bow(new_doc.lower().split())
        print(new_vec)


3.1.4 Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

常用模型：

    - tf-idf

    .. code-block:: python
    
        from gensim import models

        # 训练模型
        tfidf = models.TfidfModel(bow_corpus)

        # 转换 "system minors" 字符串
        words = "system minors".lower().split()
        print(tfidf[dictionary.doc2bow(words)])



3.2 语料(Corpora)和词空间(Vector Spaces)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




3.3 主题(Topics)和转换(Transformations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



3.4 相似性查询(Similarity Queries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



3.5 常用教程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




4.Gensim 常用 API
-----------------------------------------

