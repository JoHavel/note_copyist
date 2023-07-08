# Based on NPFL114: https://ufal.mff.cuni.cz/courses/npfl114/2122-summer
# FIXME other than shapes [n]

import tensorflow as tf
import tensorflow_probability as tfp

from parts.encoder import Encoder2Normal
from parts.decoder import Decoder

from generators.generator import Generator
from utils.my_typing import seq2str


class VAE(Generator):
    """ Variational adversarial network (it learns generating images from one-hot labels concatenated with latent space "given" by latent_prior)
        https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#12_deep_generative_models
    """
    def __init__(
            self,
            encoder: Encoder2Normal,
            decoder: Decoder,
            n_of_categories: int,
            latent_prior=None,
    ) -> None:
        super().__init__()

        self.latent_shape = decoder.inputs[0].shape[1:]

        if latent_prior is None:
            self._latent_prior = tfp.distributions.Normal(
                tf.zeros(self.latent_shape[0] - n_of_categories),
                tf.ones(self.latent_shape[0] - n_of_categories)
            )
        else:
            self._latent_prior = latent_prior

        self.encoder = encoder
        self.decoder = decoder
        self.n_of_categories = n_of_categories

    def train_step(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        images, labels = data
        with tf.GradientTape() as tape:
            # Encode to latent space distribution
            outputs = self.encoder(images, training=True)
            mean = outputs["mean"]
            sd = outputs["sd"]

            # Categorical loss
            categories = tf.one_hot(labels, self.n_of_categories)
            categorical_loss = self.encoder.compiled_loss(
                mean[..., 0:self.n_of_categories], categories
            ) + self.encoder.compiled_loss(
                sd[..., 0:self.n_of_categories], tf.zeros_like(sd[..., 0:self.n_of_categories], dtype=tf.float32)
            )

            distribution = tfp.distributions.Normal(mean[..., self.n_of_categories:], sd[..., self.n_of_categories:])

            # Decode images
            latent_space = distribution.sample()
            generated_images = self.decoder(tf.concat([categories, latent_space], axis=-1), training=True)
            reconstruction_loss = self.decoder.compiled_loss(images, generated_images)

            # Loss from "difference" from prior
            latent_loss = tf.reduce_mean(
                distribution.kl_divergence(self._latent_prior)
            )

            # Sum losses
            loss = (
                    latent_loss * tf.cast(self.latent_shape, tf.float32) +
                    reconstruction_loss * tf.cast(tf.reduce_prod(tf.shape(images)[1:]), tf.float32) +
                    categorical_loss
            )

        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.encoder.trainable_weights + self.decoder.trainable_weights),
            self.encoder.trainable_weights + self.decoder.trainable_weights)
        )

        return {"reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss, "loss": loss}

    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)

    def save_all(self, path):
        self.encoder.save(path + "e.h5")
        self.decoder.save(path + "d.h5")
        tf.io.write_file(path + "n.txt", str(self.n_of_categories))

    @staticmethod
    def load_all(path: str, latent_prior=None) -> "VAE":
        encoder = tf.keras.models.load_model(path + "e.h5", custom_objects={'Encoder2Normal': Encoder2Normal})
        decoder = tf.keras.models.load_model(path + "d.h5", custom_objects={'Decoder': Decoder})
        with open(path + "n.txt", "tr") as file:
            n_of_categories = int(file.read())
        return VAE(encoder, decoder, n_of_categories, latent_prior=latent_prior)

    @property
    def string(self):
        return f"vae_l{seq2str(self.latent_shape)};e({self.encoder.string});d({self.decoder.string})"

