import numpy as np


# ---------------
# 将数据向量化
# ---------------
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i in sequnece in enumerate(sequences):
        results[i, sequences] = 1.
        
        return results

# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)

# ---------------
# 将标签向量化
# ---------------
def to_one_hot(labels, dimension = 46):
    results = np.zeros(len(labels), dimension)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    
    return results

# one_hot_train_labels = to_one_hot(train_label)
# one_hot_test_labels = to_one_hot(test_label)

