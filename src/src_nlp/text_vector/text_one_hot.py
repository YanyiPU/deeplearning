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
        print(i, doc)
        for word in doc:
            if word in vocab:
                pos = vocab.index(word)
                onehot[i][pos] = 1
    onehot = pd.DataFrame(onehot, columns = vocab)
    onehot.to_csv(os.path.join(config.data_dir, "onehot.csv"))
    return onehot


if __name__ == "__main__":
    corpus = os.path.join(config.data_dir, "corpus.txt")
    onehot = doc2onthot_matrix(corpus)
    print(onehot)
