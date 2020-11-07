# encoding=utf-8

import jieba.posseg as psg

sent = "中文分词是文本处理不可或缺的一步!"
seg_list = psg.cut(sent)
seg_list_hmm = psg.cut(sent, HMM = True)
print(" ".join([f"{w}/{t}" for w, t in seg_list]))
print(" ".join([f"{w}/{t}" for w, t in seg_list_hmm]))


