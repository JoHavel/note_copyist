import enum
from enum import Enum

import numpy as np
import tensorflow as tf

from utils.my_typing import String

_DOWNLOADED: str = "./downloaded"
""" Directory for storing dataset data """


class DatasetPart(Enum):
    """ Possible dataset parts """
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class Dataset(String):
    """ Class representing a dataset which can be used in experiments """

    def __init__(self, shape: tuple[int, ...], X: dict[DatasetPart, tf.Tensor], y: dict[DatasetPart, tf.Tensor], string: str):
        """ Creates dataset with data `X` and labels `y` with one data having `shape` """
        self.shape: tuple[int, ...] = shape
        """ The shape of one piece of data """

        self.X: dict[DatasetPart, tf.Tensor] = X
        """ Data of the dataset """

        self.y: dict[DatasetPart, tf.Tensor] = y
        """ Labels of `self.x` """

        self.string: str = string

    @property
    def X_train(self) -> tf.Tensor:
        """ Training data """
        return self.X[DatasetPart.TRAIN]

    @property
    def X_val(self) -> tf.Tensor:
        """ Validation data """
        return self.X[DatasetPart.VAL]

    @property
    def X_test(self) -> tf.Tensor:
        """ Evaluating data """
        return self.X[DatasetPart.TEST]

    @property
    def y_train(self) -> tf.Tensor:
        """ Training labels """
        return self.y[DatasetPart.TRAIN]

    @property
    def y_val(self) -> tf.Tensor:
        """ Validation labels """
        return self.y[DatasetPart.VAL]

    @property
    def y_test(self) -> tf.Tensor:
        """ Evaluating labels """
        return self.y[DatasetPart.TEST]

    def __str__(self) -> str:
        return self.string


class DatasetModel(tf.keras.Model):
    """ tf.keras.Model returning images from dataset, always in the same ordering """

    def __init__(self, dataset: Dataset, part: DatasetPart = DatasetPart.TRAIN, *args, **kwargs):
        """ We use `part` of `dataset` for generating with `__call__()` """
        super().__init__(*args, **kwargs)
        self.i: int = 0
        """ Index of the next data to generate """
        self._X: tf.Tensor = dataset.X[part]
        """ Data from which we generate """

    def __call__(self, *args, **kwargs):
        """ TODO len(shape) > 1 """
        ans = self._X[self.i][None]
        self.i += 1
        return ans


class CategoricalDataset(Dataset):
    """ Class representing a dataset with categories which can be used in experiments """

    def __init__(self, shape: tuple[int, ...], X: dict[DatasetPart, tf.Tensor], y: dict[DatasetPart, tf.Tensor], string: str,
                 n_of_categories: int):
        """ Creates `n_of_categories`-category dataset with data `X` and labels `y` with one data having `shape` """
        super().__init__(shape, X, y, string)
        self.n_of_categories: int = n_of_categories
        """ The number of categories in this dataset """

    def one_category(self, category: int) -> Dataset:
        """ Gets only one `category` from the category dataset """
        X_new = {}
        y_new = {}
        for key in self.X.keys():
            X_new[key] = tf.boolean_mask(self.X[key], self.y[key] == category)
            y_new[key] = tf.boolean_mask(self.y[key], self.y[key] == category)
        return Dataset(self.shape, X_new, y_new, self.string)


class CategoricalDatasetModel(tf.keras.Model):
    """ tf.keras.Model returning images from dataset, always in the same ordering """
    def __init__(self, dataset: CategoricalDataset, part: DatasetPart = DatasetPart.TRAIN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ We use `part` of categorical `dataset` for generating with `__call__()` """
        self.models = [DatasetModel(dataset.one_category(cat), part) for cat in range(dataset.n_of_categories)]

    def __call__(self, category: int | tf.Tensor | np.ndarray, *args, **kwargs):
        """ TODO len(shape) > 1 """
        if isinstance(category, int):
            return self.models[category](*args, **kwargs)

        else:
            for i in range(len(self.models)):
                if category[i] == 1.0:
                    return self.models[i](*args, **kwargs)

        raise ValueError("Parameter `category` is not onehot or non-existing category was chosen.")
