import abc

import tensorflow as tf


class Generator(tf.keras.Model, abc.ABC):
    latent_shape: tuple[int]
    string: str

    def __str__(self) -> str:
        return self.string

    @abc.abstractmethod
    def save_all(self, path):
        ...

    @staticmethod
    @abc.abstractmethod
    def load_all(path: str, string: str, **kwargs) -> "Generator":
        ...


class CategoricalGenerator(Generator, abc.ABC):
    n_of_categories: int

    @staticmethod
    @abc.abstractmethod
    def load_all(path: str, string: str, n_of_categories: int, **kwargs) -> "CategoricalGenerator":
        ...
