import pandas as pd
import tensorflow as tf


# read dataset
csv_file = tf.keras.utils.get_file("heart.csv", "https://storage.googleapis.com/applied-dl/heart.csv")
df = pd.read_csv(csv_file)
print(df.head())
print(df.dtypes)

# data preprocessing
df["thal"] = pd.Categorical(df["thal"])
df["thal"] = df.thal.cat.codes
print(df.head())



target = df.pop("target")
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for features, target in dataset.take(5):
    
