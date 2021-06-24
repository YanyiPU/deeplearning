import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import kerastuner


# ===================
# Data loading
# ===================
# ------------------------------------------
# image
# ------------------------------------------
# Create a dataset-Image
# image_dataset = keras.preprocessing.image_dataset_from_directory(
#     "path/to/main_directory", 
#     batch_size = 64, 
#     image_size = (200, 200),
#     # class_name = None
# )
# Iterate over the batches yielded by the image dataset.
# for data, labels in image_dataset:
#     print(data.shape)
#     print(data.dtype)
#     print(lables.shape)
#     print(labels.dtype)


# ------------------------------------------
# text
# ------------------------------------------
# Create a dataset-Text
# text_dataset = keras.preprocessing.text_dataset_from_directory(
#     "path/to/main_directory",
#     batch_size = 64
# )
# Iterate over the batches yielded by the text dataset.
# for data, labels in text_dataset:
#     print(data.shape)
#     print(data.dtype)
#     print(labels.shape)
#     print(labels.dtype)



# ===================
# Data preprocessing
# ===================
# ------------------------------------------
# text
# ------------------------------------------
training_data = np.array([
    ["This is the 1st sample."], 
    ["And here's the 2nd sample."]
])
vectorizer = TextVectorization(output_mode = "int")
vectorizer = TextVectorization(output_mode = "binary", ngrams = 2)
vectorizer.adapt(training_data)
integer_data = vectorizer(training_data)
# print(integer_data)

# ------------------------------------------
# image
# ------------------------------------------
training_data = np.random.randint(0, 256, size = (64, 200, 200, 3)).astype("float32")
normalizer = Normalization(axis = -1)
normalizer.adapt(training_data)
normalizer_data = normalizer(training_data)
# print(normalizer_data)
# print("var: %.4f" % np.var(normalizer_data))
# print("mean: %.4f" % np.mean(normalizer_data))


training_data = np.random.randint(0, 256, size = (64, 200, 200, 3)).astype("float32")
cropper = CenterCrop(height = 150, width = 150)
scaler = Rescaling(scale = 1.0 / 255)
output_data = scaler(cropper(training_data))
# print(output_data)
# print("shape:", output_data.shape)
# print("min", np.min(output_data))
# print("max", np.max(output_data))


# ===================
# Keras functional-api
# ===================
# ------------------------------------------
# data
# ------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ------------------------------------------
# build model
# ------------------------------------------
inputs = keras.Input(shape = (28, 28))
# x = CenterCrop(height = 150, width = 150)(inputs)
x = Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
# x = layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)
# x = layers.MaxPooling2D(pool_size = (3, 3))(x)
# x = layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)
# x = layers.MaxPooling2D(pool_size = (3, 3))(x)
# x = layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)
# x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation = "relu")(x)
x = layers.Dense(128, activation = "relu")(x)
num_classes = 10
outputs = layers.Dense(num_classes, activation = "softmax")(x)
model = keras.Model(inputs, outputs, name = "wangzf")
print(model.summary())

# ------------------------------------------
# model compile
# ------------------------------------------
model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate = 1e-3),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = [keras.metrics.SparseCategoricalAccuracy(name = "acc")]
)

# ------------------------------------------
# model fit-Numpy
# ------------------------------------------
batch_size = 64
print("Fit on Numpy data.")
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = 1)
print(history.history)

# ------------------------------------------
# model fit-Dataset
# ------------------------------------------
batch_size = 64
print("Fit on Dataset.")
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
history = model.fit(dataset, epochs = 1)
print(history.history)

# ------------------------------------------
# model validate
# ------------------------------------------
batch_size = 64
print("Fit on Validate dataset.")
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs = 1, validation_data = val_dataset)
print(history.history)

# ------------------------------------------
# callbacks for checkpointing
# ------------------------------------------
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath = "path/to/my/model_{epoch}",
#         save_freq = "epoch"
#     )
# ]
# model.fit(dataset, epochs = 2, callbacks = callbacks)

# ------------------------------------------
# Tensorboard
# ------------------------------------------
# callbacks = [
#     keras.callbacks.TensorBoard(log_dir = "./logs")
# ]
# model.fit(dataset, epochs = 2, callbacks = callbacks)

# tensorboard --logdir=./logs

# ------------------------------------------
# model evaluate
# ------------------------------------------
loss, acc = model.evaluate(val_dataset)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

# or
# predictions = model.predict(val_dataset)
# print(predictions.shape)


# ------------------------------------------
# 自定义训练步骤
# ------------------------------------------
class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses = self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

# Construct and compile an instance of CustomModel
inputs = keras.Input(shape = (32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer = 'adam', loss = 'mse', metrics = [...])

# Just use `fit` as usual
model.fit(dataset, epochs = 3, callbacks = ...)

# ------------------------------------------
# 调试模型模式
# ------------------------------------------
model.compile(optimizer = "adam", loss = "mse", run_eagerly = True)

# ------------------------------------------
# GPU 分布式训练
# ------------------------------------------
# 创建一个 MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# 开启一个 strategy scope
with strategy.scope():
    model = Model()
    model.compile()

train_dataset, val_dataset, test_dataset = get_dataset()
# 在所有可用的设备上训练模型
model.fit(train_dataset, epochs = 2, validation_data = val_dataset)
# 在所有可用的设备上测试模型
model.evaluate(test_dataset)



# ------------------------------------------
# 异步处理
# ------------------------------------------
samples = np.array([
    ["This is the 1st sample."], 
    ["And here's the 2nd sample."]
])
labels = [
    [0], 
    [1]
]
# Prepare a TextVectorization layer.
vectorizer = TextVectorization(output_mode = "int")
vectorizer.adapt(samples)

# Asynchronous preprocessing: the text vectorization is part of the tf.data pipeline.
# First, create a dataset
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
# Apply text vectorization to the samples
dataset = dataset.map(lambda x, y: (vectorizer(x), y))
# Prefetch with a buffer size of 2 batches
dataset = dataset.prefetch(2)

# Our model should expect sequences of integers as inputs
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(input_dim=10, output_dim=32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)


# ------------------------------------------
# 超参数搜索调优
# ------------------------------------------
def build_model(hp):
    """返回已编译的模型

    Args:
        hp ([type]): [description]

    Returns:
        [type]: [description]
    """
    inputs = keras.Input(shape = (784,))
    x = layers.Dense(
        units = hp.Int("units", min_value = 32, max_value = 512, step = 32),
        activation = "relu",
    )(inputs)
    outputs = layers.Dense(10, activation = "softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer = keras.optimizers.Adam(hp.Choice("leraning_rate", values = [1e-2, 1e-3, 1e-4])),
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model

tuner = kerastuner.tuners.Hyperband(
    build_model,
    objective = "val_loss",
    max_epochs = 100,
    max_trials = 200,
    executions_per_trial = 2,
    directory = "my_dir"
)
tuner.search(dataset, validation_data = val_dataset)
models = tuner.get_best_models(num_models = 2)
tuner.results_summary()

