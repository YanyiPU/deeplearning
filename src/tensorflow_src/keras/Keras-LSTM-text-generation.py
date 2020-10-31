
# Keras-LSTM-text-generation

"""
# Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text starts sounds coherent.

It is recommended to run this script on GPU, as recurrent networks are quite computationally intensive.

If you try this script on new data, make sure your corpus has at least ~100k characters. ~1M is better.
"""


from __futrue__ import print_function
import keras
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Desne
from keras.layers import LSTM
from keras import optimizers, losses, metrics
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


# =======================================
# data
# =======================================
path = get_file(
   "nietzsche.txt",
   origin = "http://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
with io.open(path, encoding = "utf-8") as f:
   text = f.read().lower()
print("corpus length:", len(text))

chars = sorted(list(set(text)))
print("total chars:", len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
   sentences.append(text[i: i + maxlen])
   next_chars.append(text[i + maxlen])
print("nb sequences:", len(sentences))

print("Vectorization...")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
   for t, char in enumerate(sentence):
      x[i, t, char_indices[char]] = 1
   y[i, char_indices[next_chars[i]]] = 1


# =======================================
# Model
# =======================================
# build the model: a single LSTM
print("Build model...")
model = Sequential()
model.add(LSTM(128, input_shape = (maxlen, len(chars))))
model.add(Dense(len(chars), activation = "softmax"))

# =======================================
# Model compile
# =======================================
optimizer = optimizers.RMSprop(lr = 0.01)
model.compile(loss = "categorical_crossentropy",
            optimizer = optimizer)


def sample(preds, temperature = 1.0)
