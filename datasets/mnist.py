import tensorflow as tf
from keras.datasets import mnist

from datasets.dataset import CategoricalDataset, DatasetPart

_WIDTH = 28
""" One dimension of MNIST dataset image """


class MnistDataset(CategoricalDataset):
    """ Class encapsulating MNIST dataset """

    def __init__(self, train_len: int = 50000, multiply_of: int | None = None, image_dirs: str = None):
        """
            Gets MNIST dataset, where there is `train_len` of train data, the rest is validation.
            All images have dimensions divisible by `multiply_of`.
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        target_width = _WIDTH
        if multiply_of is not None:
            target_width += multiply_of - (_WIDTH % multiply_of)

            X_train = tf.image.pad_to_bounding_box(
                X_train[..., None], (target_width - _WIDTH)//2, (target_width - _WIDTH)//2, target_width, target_width
            )[..., 0]
            X_test = tf.image.pad_to_bounding_box(
                X_test[..., None], (target_width - _WIDTH)//2, (target_width - _WIDTH)//2, target_width, target_width
            )[..., 0]
        X_train = tf.cast(X_train, tf.float32) / 255.0
        X_test = tf.cast(X_test, tf.float32) / 255.0

        X_val = X_train[train_len:]
        X_train = X_train[:train_len]

        y_val = y_train[train_len:]
        y_train = y_train[:train_len]

        X = {DatasetPart.TRAIN: X_train, DatasetPart.VAL: X_val, DatasetPart.TEST: X_test}
        y = {DatasetPart.TRAIN: y_train, DatasetPart.VAL: y_val, DatasetPart.TEST: y_test}
        super().__init__((target_width, target_width), X, y, "mnist", 10)

