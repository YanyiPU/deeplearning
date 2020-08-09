#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===========================================================
#                        codeing
#===========================================================

#  单词级的one-hot encoding
samples = ["The cat sat on the mat.", "The dog ate my homework."]
token_index = {}
for sample in samples:
    for word in sample.split():
        token_index[word] = len(token_index) + 1

max_length = 10

results = np.zeros(shape = (len(sample), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        resultss[i, j, index] = 1.
