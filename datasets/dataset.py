import enum
from enum import Enum

import numpy as np
import tensorflow as tf


class DatasetPart(Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class Dataset:
    def __init__(self, shape: tuple[int, ...], X: dict[DatasetPart, tf.Tensor], y: dict[DatasetPart, tf.Tensor], string: str):
        self.shape = shape
        self.X = X
        self.y = y
        self.string = string

    @property
    def X_train(self):
        return self.X[DatasetPart.TRAIN]

    @property
    def X_val(self):
        return self.X[DatasetPart.VAL]

    @property
    def X_test(self):
        return self.X[DatasetPart.TEST]

    @property
    def y_train(self):
        return self.y[DatasetPart.TRAIN]

    @property
    def y_val(self):
        return self.y[DatasetPart.VAL]

    @property
    def y_test(self):
        return self.y[DatasetPart.TEST]

    def __str__(self):
        return self.string


class DatasetModel(tf.keras.Model):
    """ tf.keras.Model returning images from dataset, always in the same ordering """
    def __init__(self, dataset: Dataset, part: DatasetPart = DatasetPart.TRAIN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = 0
        self._X = dataset.X[part]

    def __call__(self, *args, **kwargs):
        """ TODO len(shape) > 1 """
        ans = self._X[self.i][None]
        self.i += 1
        return ans


class CategoricalDataset(Dataset):
    def __init__(self, shape: tuple[int, ...], X: dict[DatasetPart, tf.Tensor], y: dict[DatasetPart, tf.Tensor], string: str,
                 n_of_categories: int):
        super().__init__(shape, X, y, string)
        self.n_of_categories = n_of_categories

    def one_category(self, category: int) -> Dataset:
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
