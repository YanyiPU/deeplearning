
NLP Pipeline
============================================

1.构建 NLP Pipeline
--------------------------------------------

    .. image:: ../../../images/London.gif

1.0 数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important:: 

    London is the capital and most populous city of England and the United 
    Kingdom. Standing on the River Thames in the south east of the island of Great 
    Britain, London has been a major settlement for two millennia. It was founded 
    by the Romans, who named it Londinium.
    
    (Source: `Wikipedia article "London" <https://en.wikipedia.org/wiki/London>`_ )

上面这段文本包含了一些有用的事实，我们需要让计算机能够理解以下内容：

    - London is a city
    - London is located in England
    - London was settled by Romans
    - and so on

1.1 Step 1: 分句(Sentence Segmentation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NLP Pipeline 的第一步就是 **将文本分割为句子**:

        - 1."London is the capital and most populous city of England and the United Kingdom."
        - 2."Standing on the River Thames in the south east of the island of Great Britain, London has been a major settlement for two millennia."
        - 3."It was founded by the Romans, who named it Londinium."

1.2 Step 2: 分词(Word Tokenization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - 句子:

        .. image:: ../../../images/London_sentences.png

    - 分词(tokenization):

        .. image:: ../../../images/London_word.png

    .. note:: 

        Tokenization 对于英文来说非常容易，可以根据单词之间的空格对文本句子进行分割。
        并且，也可以将标点符号也看成一种分割符，因为标点符号也是有意义的。

1.3 词性标注(Predicting Parts of Speech for Each Token)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    通过词性标注，可以知道每个词在句子中的角色，即词性，词性包括名词(noun)、
    动词(verb)、形容词(adjective)等等，这样可以帮助计算机理解句子的意思.

    通过将每个单词(以及其周围的一些额外单词用于上下文)输入到预先训练的词性分
    类模型中进行词性标注. 词性分类模型完全基于统计数据根据以前见过的相似句子
    和单词来猜词性。

    - 词性标注过程

        .. image:: ../../../images/sentence1.png

    - 词性标注结果

        .. image:: ../../../images/sentence2.png
        




    通过词性标注后的信息，可以收集到一些关于句子的基本含义了，例如：可以看到句子中看到有"London"、
    和 "captial" 两个名词(Noun)，因此该句子可能在谈论 "London".


1.4 (Text Lemmatization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    在 NLP 中，称此过程为 词性限制(lemmatization)，即找出句子中每个单词的最基
    本形式或词缀(lemma)。

    - 示例:

        在自然语言中，单词或词会以不同的形式同时出现：

            - I had a **pony**.
            - I had two **ponies**.

        上面两个句子中都在表达 **pony**，但是两个句子中使用了 **pony** 不同的词缀。

        在计算机处理文本时，了解每个单词的基本形式很有帮助，这样就可以知道两个句子都在
        表达相同的概念，否则，字符串 "pony" 和 "ponies" 对于计算机来说就像两个完全
        不同的词.
        
        同样对于动词(verb)道理也一样:

            - I had two ponies.
            - I have two pony.

    - London 句子示例:

        .. image:: ../../../images/sentence3.png

1.5 识别停用词(Identifying Stop Words)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    停用词的识别主要是用来考虑句子中每个单词的重要性。
    
    自然语言汇总有很多填充词，例如: "and"，"the"，"a"，这些词经常出现。
    在对文本进行统计时，这些填充词会给文本数据带来很多噪音，因为他们比其他
    单词更频繁地出现。一些 NLP Pipeline 会将它们标记为 **停用词(Stop Word)**。
    这些停用词需要在进行任何统计分析之前将其过滤掉。

    - London 示例:

        .. image:: ../../../images/sentence4.png

    .. note:: 
    
        - 通常仅通过检查已知停用词的硬编码列表即可识别停用词.
        - 没有适用于所有应用程序的标准停用词列表，要忽略的单词列表可能会因应用程序而异.
        -   例如：如果要构建摇滚乐队搜索引擎，则要确保不要过滤到单词 "The"。因为，"The" 一词
            会经常出现在乐队的名字中，比如 1980年代的注明摇滚乐队 "The The!".

1.6 句法依赖分析(Dependency Parsing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






1.7 命名实体识别
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.8 共指解析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




2.构建 Python NLP Pipeline
--------------------------------------------

    .. image:: ../../../images/NLP_Pipeline.png

3.工具
----------------------------------------------

    - 参考文章

        - `NLP is Fun! <https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e>`_ 
        - `Part2 <https://medium.com/@ageitgey/text-classification-is-your-new-secret-weapon-7ca4fad15788>`_ 
        - `Part3 <https://medium.com/@ageitgey/natural-language-processing-is-fun-part-3-explaining-model-predictions-486d8616813c>`_ 
        - `Part4 <https://medium.com/@ageitgey/deepfaking-the-news-with-nlp-and-transformer-models-5e057ebd697d>`_ 
        - `Part5 <https://medium.com/@ageitgey/build-your-own-google-translate-quality-machine-translation-system-d7dc274bd476>`_ 

    - Python NLP 库

        - `NeuralCoref <https://github.com/huggingface/neuralcoref>`_ 
        - `spaCy <https://spacy.io/>`_ 
        - `textacy <https://textacy.readthedocs.io/en/stable/>`_ 