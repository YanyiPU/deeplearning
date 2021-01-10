
NLP--fastText
============================


1.fastText 简介
----------------------------

    - 官网：https://fasttext.cc/

1.1 快速开始
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.1.1 fastText 是什么
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    fastText is a library for efficient learning of **word representations** and **sentence classification**.

1.1.2 fastText 环境依赖
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - 计算机系统

        - macOS
        - Linux

    - C++11 编译器

        - (gcc-4.6.3 or newer) or (clang-3.3 or newer)
        - make

    - Python 依赖

        - >=python 2.6
        - numpy
        - scipy

1.1.3 fastText 工具库构建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.构建 fastText 为一个命令行工具(CLT)

    .. code-block:: shell

        $ git clone https://github.com/facebookresearch/fastText.git
        $ cd fastText
        $ make
    
    或者:

    .. code-block:: shell

        $ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
        $ unzip v0.9.2.zip
        $ cd fastText-0.9.2
        $ make

2.构建 fastText 为一个 Python 模块

    .. code-block:: shell

        $ git clone https://github.com/facebookresearch/fastText.git
        $ cd fastText
        $ sudo pip install .
        # or
        $ sudp python setup.py install
    
    或者:

    .. code-block:: shell

        $ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
        $ unzip v0.9.2.zip
        $ cd fastText-0.9.2
        $ pip install

    .. code-block:: python

        >>> import fasttext
        >>>

3.获取帮助:

    .. code-block:: shell
    
        ./fasttext

    .. code-block:: python

        >>> import fasttext
        >>> help(fasttext.FastText)

2.使用 fastText 进行文本分类
--------------------------------------

    文本分类可以应用在许多方面：

        - spam detection
        - sentiment analysis
        - smart replies

2.1 准备文本数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 数据来源: https://cooking.stackexchange.com/

    - 数据描述: 

        - building a classifier to automatically recognize the topic of a stackexchange question about cooking

    - 数据下载

        .. code-block:: shell

            $ wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz
            $ tar xvzf cooking.stackexchange.tar.gz
            $ head cooking.stackexchange.txt
            $ wc cooking.stackexchange.txt

    - 数据格式预览

        ============================================================================== ==============================================================================
        Label                                                                           document
        ============================================================================== ==============================================================================
        __label__sauce __label__cheese                                                 How much does potato starch affect a cheese sauce recipe?
        __label__food-safety __label__acidity                                          Dangerous pathogens capable of growing in acidic environments
        __label__cast-iron __label__stove                                              How do I cover up the white spots on my cast iron stove?
        __label__restaurant                                                            Michelin Three Star Restaurant; but if the chef is not there
        __label__knife-skills __label__dicing                                          Without knife skills, how can I quickly and accurately dice vegetables?
        __label__storage-method __label__equipment __label__bread                      What's the purpose of a bread box?
        __label__baking __label__food-safety __label__substitutions __label__peanuts   how to seperate peanut oil from roasted peanuts at home?
        __label__chocolate                                                             American equivalent for British chocolate terms
        __label__baking __label__oven __label__convection                              Fan bake vs bake
        __label__sauce __label__storage-lifetime __label__acidity __label__mayonnaise  Regulation and balancing of readymade packed mayonnaise and other sauces
        ============================================================================== ==============================================================================

    - 数据集分割

        - Training dataset

            .. code-block:: shell

                $ head -n 12404 cooking.stackexchange.txt > cooking.train
                $ wc cooking.train

        - validation dataset

            .. code-block:: shell

                $ tail -n 3000 cooking.stackexchange.txt > cooking.valid
                $ wc cooking.valid

2.2 构建分类器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 基本模型

    .. code-block:: python
    
        import fasttext
    
        # 模型训练    
        model = fasttext.train_supervised(input = "cooking.train")
    
        # 模型保存
        model.save_model("model_cooking.bin")

        # 模型测试
        model.predict("Which baking dish is best to bake a banana bread ?")
        model.predict("Why not put knives in the dishwater?")
        model.test("cooking.valid")
        model.test("cooking.valid", k = 5)

- precision 和 recall

        # Top 5 预测标签，用来计算 precision 和 recall
        model.predict("Why not put knives in the dishwater?", k = 5)

- 增强模型预测能力

    - (2)数据预处理

        - 将单词中的大写字母转换为小写字母
        - 处理标点符号

        .. code-block:: shell

            $ cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
            $ head -n 12404 cooking.preprocessed.txt > cooking_preprocessed.train
            $ tail -n 3000 cooking.preprocessed.txt > cooking_preprocessed.valid

        .. code-block:: python

            import fasttext

            model = fasttext.train_supervised(input = "cooking_preprocessed.train")
            model.test("cooking_preprocessed.valid")

    - (2)增多 epochs

        .. code-block:: python

            import fasttext

            model = fasttext.train_supervised(input = "cooking.train", epoch = 25)
            model.test("cooking.valid")

    - (3)增大 learning_rate

        .. code-block:: python

            import fasttext

            model = fasttext.train_supervised(input = "cooking.train", lr = 1.0)
            model.test("cooking.valid")

        .. code-block:: python
        
            import fasttext

            model = fasttext.train_supervised(input = "cooking.train", lr = 1.0, epoch = 25)
            model.test("cooking.valid")

    
    - (4)word n-grams

        .. code-block:: python

            model = fasttext.train_supervised(
                input = "cooking.train", 
                lr = 1.0, 
                epoch = 25, 
                wordNgrams = 2
            )
            model.test("cooking.valid")

- Bigram


- Scaling thing up

    .. code-block:: python

        model = fasttext.train_supervised(
            input = "cooking.train", 
            lr = 1.0, 
            epoch = 25, 
            wordNgrams = 2, 
            bucket = 200000, 
            dim = 50,
            loss = "hs"
        )

- 多标签分类(Multi-label classification)

    .. code-block:: python

        import fasttext

        model = fasttext.train_supervised(
            input = "cooking.train", 
            lr = 0.5, 
            epoch = 25, 
            wordNgrams = 2, 
            bucket = 200000, 
            dim = 50, 
            loss = "ova"
        )

        model.predict(
            "Which baking dish is best to bake a banana bread ?", 
            k = -1,
            threshold = 0.5
        )
        model.test("cooking.valid", k = -1)


3.使用 fastText 进行词表示
---------------------------------------

