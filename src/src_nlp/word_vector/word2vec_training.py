# -*- coding: utf-8 -*-
import os
from gensim.corpora import WikiCorpus
import codecs
import jieba
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import opencc


# 项目目录
project_path = os.path.abspath(".")
data_path = "/Users/zfwang/project/machinelearning/deeplearning/data/NLP_data/word2vec_data"

# 维基百科中文语料
corpus_path = os.path.join(data_path, "zhwiki-latest-pages-articles.xml.bz2")
# Gensim数据提取
extracted_corpus_path = os.path.join(data_path, "wiki-zh-article.txt")
# 繁体中文转换为简体中文
extracted_zhs_corpus_path = os.path.join(data_path, "wiki-zh-article-zhs.txt")
# 分词
cuted_word_path = os.path.join(data_path, "wiki-zh-words.txt")
# 模型
model_output_1 = os.path.join(project_path, "src/nlp_src/models/wiki-zh-model")
model_output_2 = os.path.join(project_path, "src/nlp_src/models/wiki-zh-vector")




if __name__ == "__main__":
    # ----------------------------------------
    # 1.数据预处理
    # ----------------------------------------
    #TODO 判断目录中是否存在相关数据文件
    # space = " "
    # with open(extracted_corpus_path, "w", encoding = "utf8") as f:
    #     wiki = WikiCorpus(corpus_path, lemmatize = False, dictionary = {})
    #     for text in wiki.get_texts():
    #         print(text)
    #         f.write(space.join(text) + "\n")
    # print("Finished Saved.")
    # ----------------------------------------
    # 2.繁体字处理
    # ----------------------------------------
    # t2s_converter = opencc.OpenCC("t2s.json")
    # with open(extracted_corpus_path, "r", encoding = "utf8") as f1:
    #     with open(extracted_zhs_corpus_path, "w", encoding = "utf8") as f2:
    #         extracted_zhs_corpus = t2s_converter.convert(f1)
    #         f2.write(extracted_zhs_corpus)
    # print("Finished Converter.")
    # ----------------------------------------
    # 3.分词
    # ----------------------------------------
    # descsFile = codecs.open(extracted_zhs_corpus_path, "rb", encoding = "utf-8")
    # i = 0
    # with open(cuted_word_path, "w", encoding = "utf-8") as f:
    #     for line in descsFile:
    #         i += 1
    #         if i % 10000 == 0:
    #             print(i)
    #         line = line.strip()
    #         words = jieba.cut(line)
    #         for word in words:
    #             f.write(word + " ")
    #         f.write("\n")
    # ----------------------------------------
    # 4.运行 word2vec 训练模型
    # ----------------------------------------
    # model = Word2Vec(
    #     LineSentence(cuted_word_path), 
    #     size = 400, 
    #     window = 5, 
    #     min_count = 5, 
    #     workers = multiprocessing.cpu_count()
    # )
    # model.save(model_output_1)
    # model.save_word2vec_format(model_output_2, binary = False)
    # ----------------------------------------
    # 5.模型测试
    # ----------------------------------------
    # model = Word2Vec(model_output_1)
    # # model = Word2Vec.load_word2vec_format(model_output_2, binary = False)
    # res = model.most_similar("时间")
    # print(res)
