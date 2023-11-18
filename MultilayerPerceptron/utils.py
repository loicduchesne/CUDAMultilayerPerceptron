import numpy as np

def one_hot_encoding(y):
    unique_classes = np.unique(y)
    Y = np.zeros((y.shape[0], len(unique_classes)))

    for i, label in enumerate(unique_classes):
        rows = np.where(y == label)[0]
        Y[rows, i] = 1
    return Y

def one_hot_decoding(Y):
    return np.argmax(Y, axis=1) + 1

def prob_to_class(Y):
    return np.rint(Y)

def accuracy(arr1, arr2):
    arr2.shape = arr1.shape
    accuracy = np.mean(np.all(arr1 == arr2, axis=1))
    return accuracy