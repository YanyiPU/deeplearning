import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


model = keras.Sequential()
model.add(layers.Dense(2, activation = "relu"))
model.add(layers.Dense(3, activation = "relu"))
model.add(layers.Dense(4))
print(model.layers)

model.pop()
print(len(model.layers))
