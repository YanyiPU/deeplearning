
NLP
==============

1.NLP 知识框架
------------------------

   - 什么是 NLP?

      - NLP 的概念

         - 1.NLP(Natural Language Processing, 自然语言处理)是计算机领域以及人工智能领域的一个重要的研究方向，
           它研究用计算机来处理、理解以及运用人类语言，达到人与计算机之间的有效通讯。从建模的角度看，为了方便计算机处理。

         - 2.NLP 研究表示语言能力、语言应用的模型，通过建立计算机框架来实现这样的语言模型，并且不断完善这样的语言模型，
           还需要根据该语言模型来设计各种使用的系统，并且探讨这些实用技术的评测技术。

         - 3.从自然语言的角度出发，NLP 结构如下: 

            - **NLP**

               - **自然语言理解**

                  - **音系学**: 指代语言中发音的系统化组织
                  - **词态学**: 研究单词构成以及相互之间的关系
                  - **句法学**: 给定文本的那本分是语法正确的
                  - **语义学**: 给定文本的含义是什么
                  - **语用学**: 文本的目的是什么

               - **自然语言生成** (Natural Language Generation, NLG): 从结构化数据中以读取的方式自动生成文本。该过程主要
                 包含三个阶段: 文本规划(完成机构化数据中的基础内容规划)、语句规划(从结构化数据中组合语句来表达信息流)、
                 实现(产生语法通顺的语句来表达文本)

                  - **自然语言文本**

      - NLP 的研究任务

         - **机器翻译**: 计算机具备将一种语言翻译成另一种语言的能力

         - **情感分析**: 计算机能够判断用户评论是否积极

         - **智能问答**: 计算机能够正确回答输入的问题

         - **文摘生成**: 计算机能够准确归纳、总结并产生文本摘要

         - **文本分类**: 计算机能够采集各种文章，进行主题分析，从而进行自动分类

         - **舆论分析**: 计算机能够判断目前舆论的导向

         - **知识图谱**: 知识点相互连接而成的语义网路

   - NLP 相关知识的构成:

      - 基本术语

         - **分词(segment)**

            - 分词常用的方法是基于字典的最长串匹配，但是歧义分词很难

         - **词性标注(part-of-speech tagging)**

            - 词性一般是指动词(noun)、名词(verb)、形容词(adjective)等

            - 标注的目的是表征词的一种隐藏状态，隐藏状态构成的转移就构成了状态转义序列

         - **命名实体识别(NER, Named Entity Recognition)**

            - 命名实体是指从文本中识别具有特定类别的实体(通常是名词)

         - **句法分析(syntax parsing)**

            - 句法分析是一种基于规则的专家系统。句法分析的目的是解析句子中各个成分的依赖关系。
              所以，往往最终生成的结果是一棵句法分析树。句法分析可以解决传统词袋模型不考虑上下文的问题

         - **指代消解(anaphora resolution)**

            - 中文中带刺出现的频率很高，它的作用是用来表征前文出现过的人名、地名等

         - **情感识别(emotion recognition)**

            - 情感识别本质上是分类问题.通常可以基于词袋模型+分类器，或者现在流行的词向量模型+RNN

         - **纠错(correction)**

            - 基于 N-Gram 进行纠错、通过字典树纠错、有限状态机纠错

         - **问答系统(QA system)**

            - 问答系统往往需要语言识别、合成、自然语言理解、知识图谱等多项技术的配合才会实现得比较好

      - 知识结构

         - 句法语义分析

         - 关键词抽取

         - 文本挖掘

         - 机器翻译

         - 信息检索

         - 问答系统

         - 对话系统

   - NLP 的三个层面:

      - 词法分析

         - ``分词``
         - ``词性标注``

      - 句法分析

         - 短语结构句法体系
         - 依存结构句法体系
         - 深层文法句法分析

      -语义分析

         - 语义角色标注(semantic role labeling)

   - NLP 常用语料库:

      - 中文

         - `中文维基百科 <https://dumps.wikimedia.org/zhwiki/>`_ 

         - `搜狗新闻语料库 <http://download.labs.sogou.com/resource/ca.php>`_ 

         - `IMDB 情感分析语料库 <https://www.kaggle.com/tmdb/tmdb-moive-metadata>`_ 

         - 豆瓣读书

         - 邮件相关

      - 英文
   
   - NLP 实现工具:

      - numpy

      - 正则表达式


2.分词(segment)
--------------------------------------

2.1 中文分词
~~~~~~~~~~~~~~~~~~~~~~

2.1.1 中文分词方法
^^^^^^^^^^^^^^^^^^^^^^

规则分词
''''''''''''''''''''''

   基于规则的分词是一种机械的分词方法，主要通过维护词典，在切分语句时，将语句的每个字符串与词汇表中的词逐一进行匹配，找到则切分，否则不予切分；


正向最大匹配法
''''''''''''''''''''''

.. code:: python

   class MM(object):
       def __init__(self):
           self.window_size = 3

       def cut(self, text):
           result = []
           index = 0
           text_length = len(text)
           dic = ["研究", "研究生", "生命", "命", "的", "起源"]
           while text_length > index:
               for size in range(self.window_size + index, index, -1):
                   piece = text[index:size]
                   if piece in dic:
                       index = size - 1
                       break
               index = index + 1
               result.append(piece + "----")
           print(result)

   if __name__ == "__main__":
       text = "研究生命的起源"
       tokenizer = MM()
       print(tokenizer.cut(text))

双向最大匹配法
''''''''''''''''''''''


2.1.2 统计分词
^^^^^^^^^^^^^^


2.1.3 混合分词
^^^^^^^^^^^^^^


1.2 中文分词工具
~~~~~~~~~~~~~~~~

   -  jieba

   -  精确模式

      -  试图将句子最精确地切开，适合文本分析；

   -  全模式

      -  把句子中的所有可以成词的词语都扫描出来，速度非常快，但是不能解决歧义；

   -  搜索引擎模式

      -  在精确模式的基础上，对长词进行再次切分，提高召回率，适合用于搜索引擎分词；

-  常用分词库

   -  StanfordNLP

   -  哈工大语言云

   -  庖丁解牛分词

   -  盘古分词 (ICTCLAS, 中科院汉语词法分析系统)

   -  IKAnalyzer（Luence项目下，基于java）

   -  FudanNLP（复旦大学）

   -  中文分词工具

   -  ``Ansj``

   -  盘古分词

   -  ``jieba``

安装:

    .. code-block:: shell

        pip install jieba

三种分词模式：

    .. code-block:: python

        import jieba

        sent = "中文分词是文本处理不可或缺的一步！"

        # 精确模式
        seg_list = jieba.cut(sent, cut_all = False)

        # 全模式
        seg_list = jieba.cut(sent, cut_all = True)

        # 搜索引擎模式
        seg_list = jieba.cut_for_search(sent)

加载自定义词典

    .. code-block:: python

        jieba.load_userdict("./data/user_dict.utf8")

示例：高频词提取

    .. code-block:: python

        # 读取数据
        def get_content(path):
            with open(path, "r", encoding = "gbk", errors = "ignore") as file:
                content = ""
                for line in file:
                    line = line.strip()
                    content += line
                return content

        # 定义高频词统计函数
        def get_TF(words, topK = 10):
            tf_dic = {}
            for w in words:
                tf_dic[w] = tf_dic.get(w, 0) + 1
                return sorted(tf_dic.items(), key = lambda x: x[1], reverse = True)[:topK]

        # 调用停用词典，过滤停用词
        def stop_words(path):
            with open(path) as file:
                return [line.strip() for line in file]

        # 加载自定义领域词典提高分词效果


        def main():
            import glob
            import random
            import jieba
            files = glob.glob("./data/news/C000013/*.txt")
            corpus = [get_content(x) for x in files]
            sample_inx = random.randint(0, len(corpus))
            split_words = [x for x in jieba.cut(corpus[sample_inx]) if x not in stop_words("./data/stop_words.utf8")]
            print("样本之一：" + corpus[sample_inx])
            print("样本分词效果：" + "/ ".join(split_words))
            print("样本的topK(10)词：" + str(get_TF(split_words)))

2.2 英文分词
~~~~~~~~~~~~~~~~~~~~~~




3.词性标注
------------------------

2.1 词性标注
~~~~~~~~~~~~

   -  词性标注是在给定句子中判断每个词的语法范畴，确定其词性并加以标注的过程；

   -  词性标注最简单的方法是从预料库中统计每个词对应的高频词性，将其作为默认的词性；


2.1.1 词性标注规范
^^^^^^^^^^^^^^^^^^

-  北大词性标注集

-  宾州词性标注集


2.1.2 jieba 分词中的词性标注
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  类似分词流程，jieba
   的词性标注同样是结合规则和统计的方式，具体为在词性标注的过程中，词典匹配和
   HMM 共同作用；

-  词性标注流程：



4.命名实体识别
----------------------


5.关键词提取算法
----------------------


6.句法分析
----------------------


7.自然语言处理与词向量
----------------------

   自然语言处理主要研究使用计算机来处理、理解以及运用人类语言的各种理论和方法，属于人工智能的一个重要研究方向；


5.1 词汇表征
~~~~~~~~~~~~


5.2 词向量与语言模型
~~~~~~~~~~~~~~~~~~~~


8.word2vec 词向量
-----------------------------

从深度学习的角度看，假设将 NLP
的语言模型看作是一个监督学习问题：给定上下文词 :math:`X`\ ，输出中间词
:math:`Y`\ ；或者给定中间词 :math:`X`\ ，输出上下文词
:math:`Y`\ 。基于输入 :math:`X` 和输出 :math:`Y`
之间的映射便是语言模型。这样的一个语言模型的目的便是检查 :math:`X` 和
:math:`Y` 放在一起是否符合自然语言规则，更通俗一点就是 :math:`X` 和
:math:`Y` 放在一起是不是人话。

所以，基于监督学习的思想，word2vec
便是一种基于神经网络训练的自然语言模型。word2vec 是谷歌于 2013
年提出的一种 NLP
工具，其特点就是将词汇进行向量化，这样就可以定量的分析和挖掘词汇之间的联系。因而
word2vec
也是词嵌入表征的一种，只不过这种向量表征需要经过神经网络训练得到。

word2vec 训练神经网路得到的一个关于输入 :math:`X` 和输出 :math:`Y`
之间的语言模型，关注的重点并不是说要把这个模型训练的有多好，而是要获取训练好的神经网络权重，这个权重就是我们要拿来对输入词汇
:math:`X` 的向量化表示。一旦拿到了训练预料所有词汇的词向量，接下来开展
NLP 分析工作就相对容易一些。


9.词向量的训练
--------------------

