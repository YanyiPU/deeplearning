# -*- coding: utf-8 -*-
import jieba.posseg as psg


# data
sent = "中文分词是文本处理不可或缺的一步!"

# 非 HMM 词性标注
seg_list = psg.cut(sent)
print(" ".join([f"{w}/{t}" for w, t in seg_list]))

# HMM 词性标注
seg_list_hmm = psg.cut(sent, HMM = True)
print(" ".join([f"{w}/{t}" for w, t in seg_list_hmm]))
