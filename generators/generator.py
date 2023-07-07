import abc

import tensorflow as tf
import tensorflow_probability as tfp

from utils.my_typing import String


class Generator(tf.keras.Model, String, abc.ABC):
    latent_shape: tuple[int]
    latent_prior: tfp.distributions.Distribution

    @abc.abstractmethod
    def save_all(self, path):
        ...

    @staticmethod
    @abc.abstractmethod
    def load_all(path: str, latent_prior: tfp.distributions.Distribution | None) -> "Generator":
        ...

