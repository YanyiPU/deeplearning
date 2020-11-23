
Python Natural Language Processing
=======================================

0.Parpre
----------------------------------------------------





1.Introduction
----------------------------------------------------

    - Understanding NLP

    - Understanding basic applications

    - Understanding advance applications

    - Advantages of togetherness-NLP and Python

    - Environment setup for NLTK

    - Tips for readers

1.1 Understanding NLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 自然语言

    - 自然语言处理

        - Natural language processing is the ability of computational technologies and/or computational linguistics to process human natural language

        - Natural language processing is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages

        - Natural language processing can be defined as the automatic (or semi- automatic) processing of human natural language


- NLP

    - Person Skill set

    - Fundamental

    - Programming

    - Statics

    - Corpus Analysis

    - Computational linguistics

    - Machine Learning

    - Toolbox

        - Scikit learn

        - TensorFlow

        - NoSql Databased

        - Apache Hadoop



.. code-block:: 

    # How build an intelligent system using concepts of NLP?

    Understand the problem statement
    Collect dataset/corpus
    Analyze dataset/corpus
    Preprocessing of dataset
    Feature engineering
    Decide the computational techniques such as Machine Learning, Rule Based and so on
    Apply computational techniques
    Test and evaluate result of system
    Tune parameters for optimization
    Continue till you will get a satisfactory result


1.2 Understanding basic applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 语音识别系统(Speech recognition system)

    - 问答系统(Question answering system)

    - 翻译系统(Translation from one specific language to another specific language)

    - 文本摘要(Text summarization)

    - Sentiment Analysis

    - Template-based chatboot

    - Text classification

    - Topic segmentation


1.3 Understanding advanced applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




1.4 Environment setup for NLTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `NLPython <https://github.com/jalajthanaki/NLPython>`_ 


1.5 Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - praticate





2.Practical Understanding of a Corpus and Dataset
----------------------------------------------------

2.0 内容概要
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 什么是语料？(What is corpus?)

    - 为什么需要语料？(Why do we need corpus?)

    - 明白语料分析(Understanding corpus analysis)

    - 明白数据属性类型(Understanding types of data attributes)

    - 探索数据的不同文件格式(Exploring different file formats of datasets)

    - 获取免费语料的资源(Resources for access free corpus)

    - 为 NLP 应用准备数据集(Preparing datasets for NLP applications)

    - 开发网页抓取应用(Developing the web scrapping application)

2.1 什么是语料？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一般认识：

    - NLP 相关应用的构建需要大量的数据集，即语料(corpus, corpora).

学术定义：

    - 语料库是存储在计算机中的书面或口语自然语言材料的集合，更准确地说，
      语料库是用于语言分析和语料库分析的真实语言的系统计算机化集合。多个语料称为语料库。

    - 语料库是作为构建 NLP 应用的输入用的，NLP 应用可以使用单一的语料库，也可以使用多个语料库作为输入。

语料库的作用：

    - 使用语料库可以对数据进行统计分析，如频率分布分析，单词共现(co-occurrences of words)等。

    - 为 NLP 应用定义和验证语言学规则，例如：语法校正系统

    - 基于语言的用法，定义一些特殊的语言学规则。在基于规则系统的基础上，利用语料库定义语言学规则并验证规则。

语料库数据的格式：

    - 文本数据(Text data)，书面、书写材料

        - 文本数据的形式、获取方式:

            - 新闻文章

            - 书籍

            - 数字图书馆

            - 电子邮件信息

            - 网页

            - 博客

            - 等等

    - 语音数据(Speech data)，口语材料



语料的形式：

    - Monolingual corpus: 只有一种语言

    - Bilingual corpus: 包含两种语言

    - Multilingual corpus: 包含超过一种语言

- 常用语料：

    - Google Books Ngram corpus

    - Brown corpus
    
    - American National corpus


2.2 为什么需要语料？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





2.3 语料分析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    语料分析是一种以真实的自然语言交流为基础，对语言概念进行深入研究的方法。
    在构建 NLP 应用前都需要一些语料分析，以更好地理解语料数据。

- 文本语料分析

    - 对数进行统计分析、处理、泛化

        - 语料中有多少个不同的单词

        - 语料中每个单词的词频

        - 消除语料中的噪音

- 语音语料分析

    - 对每个数据实例的语音理解进行分析(语音分析，phonetic analysis)

    - 对话分析(conversation analysis)


``nltk`` 中包含的语料类型：

    - Isolate corpus

        - 文本、自然语言的集合

        - 例如：gutenberg, webtext
    
    - Categorized corpus

        - 被分组为不同类别的文本集合

        - 例如：brown

    - Overlapping corpus

        - 被分组为不同类别的文本集合, 不同类别之间有重叠

        - 例如：reuters

    - Temporal corpus

        - 一段时间内自然语言用法的集合

        - 例如：inaugural address



2.4 数据属性类型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Data attributes

    - Categorical(Qualitative)

        - Ordinal

        - Nominal

    - Numeric(Quantitative)

        - Continuous

        - Discrete

2.5 数据文件格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - ``.txt``

    - ``.csv``

    - ``.tsv``

    - ``.xml``

    - ``.json``

    - ``.LibSVM``

.. note:: 

    数据中存储了可以直接输入到机器学习算法中的特征数据


2.6 获取免费语料的资源
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``nltk`` 库提供的内置语料

    .. code-block:: python
    
        import nltk.corpus
        dir(nltk.corpus)
        print(dir(nltk.corpus))

- Big Data: 33 Brilliant and Free Data Sources for 2016, Bernard Marr(英文)
    
    - https://www.forbes.com/sites/bernardmarr/2016/02/12/big-data-35-brilliant-and-free-data-sources-for-2016/#53369cd5b54d


2.7 为 NLP 应用准备数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.7.1 数据选取(获取)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

数据选取前的准备：

    1. 考虑每个 NLP 应用的所需数据集类型
    2. 考虑每个 NLP 应用的最终结果
    3. 理解要构建的 NLP 应用的问题

数据常用来源：

    - https://github.com/caesar0301/awesome-public-datasets
    - https://www.kaggle.com/datasets
    - https://www.reddit.com/r/datasets/
    - 搜索引擎
    - Python 爬虫

2.7.2 数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    1. 数据格式转换(Formatting)
        - 将数据转换为自己舒服的格式，例如：JSON, CSV...
    2. 数据清洗
        - 缺失值处理
            - 删除
            - 标记
            - 填充(临近值)
        - 数据属性处理
            - 删除不必要的数据属性
        - 无用数据处理
            - 数学公式等
    3. 数据采样
        - 指出可用的数据属性
        - 识别可以导出的数据属性
        - 识别数据中的重要数据属性


2.7.3 数据转换(特征工程)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - 应用特征工程技术将文本类型数据转换为机器可以理解的数值型数据，具体如下：
        - encoding
        - vectorization
    - 基本特征工程和 NLP 算法(第5章)
    - 高阶特征工程和 NLP 算法(第6章)


2.8 开发网页抓取应用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests
    from bs4 import BeautifulSoup

    def Get_the_page_by_beautifulsoup():
        page = requests.get("https://simplifydatascience.wordpress.com/about/")
