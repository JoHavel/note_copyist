import abc

import tensorflow as tf
import tensorflow_probability as tfp

from utils.my_typing import String


class Generator(tf.keras.Model, String, abc.ABC):
    """ Abstract class representing Generative neural network """

    latent_shape: tuple[int]
    """ Shape of the latent space """
    latent_prior: tfp.distributions.Distribution
    """ Desired distribution on the latent space """

    @abc.abstractmethod
    def save_all(self, path):
        """ Saves all models required by Generator """
        ...

    @staticmethod
    @abc.abstractmethod
    def load_all(path: str, latent_prior: tfp.distributions.Distribution | None = None) -> "Generator":
        """ Loads all models and information need to construct `Generator` and construct it """
        ...

