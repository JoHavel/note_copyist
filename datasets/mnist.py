import tensorflow as tf
from keras.datasets import mnist

from datasets.dataset import CategoricalDataset, DatasetPart


class MnistDataset(CategoricalDataset):
    def __init__(self, train_len: int = 50000):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = tf.cast(X_train, tf.float32) / 255.0
        X_test = tf.cast(X_test, tf.float32) / 255.0

        X_val = X_train[train_len:]
        X_train = X_train[:train_len]

        y_val = y_train[train_len:]
        y_train = y_train[:train_len]

        X = {DatasetPart.TRAIN: X_train, DatasetPart.VAL: X_val, DatasetPart.TEST: X_test}
        y = {DatasetPart.TRAIN: y_train, DatasetPart.VAL: y_val, DatasetPart.TEST: y_test}
        super().__init__((28, 28), X, y, 10)

