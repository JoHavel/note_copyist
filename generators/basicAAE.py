# Based on basicGAN, basicVAE,
# https://medium.com/vitrox-publication/adversarial-auto-encoder-aae-a3fc86f71758,
# and https://arxiv.org/pdf/1511.05644.pdf.

import tensorflow as tf
import tensorflow_probability as tfp

from generators.generator import Generator

from parts.discriminator import Discriminator
from parts.decoder import Decoder
from parts.encoder import Encoder2Normal
from utils.my_typing import seq2str


class AAE(Generator):
    """ Adversarial auto encoder (it learns generating images from latent space "given" by latent_prior)
        https://medium.com/vitrox-publication/adversarial-auto-encoder-aae-a3fc86f71758,
        https://arxiv.org/pdf/1511.05644.pdf.
    """
    def __init__(
            self,
            encoder: Encoder2Normal,
            decoder: Decoder,
            discriminator: Discriminator,
            latent_prior=None,
    ) -> None:
        super().__init__()

        self.latent_shape = decoder.inputs[0].shape[1:]

        if latent_prior is None:
            self._latent_prior = tfp.distributions.Normal(tf.zeros(self.latent_shape), tf.ones(self.latent_shape))
        else:
            self._latent_prior = latent_prior

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def train_step(self, images: tf.Tensor) -> dict[str, tf.Tensor]:
        # Generator
        with tf.GradientTape() as tape:
            # Encode to latent space distribution
            outputs = self.encoder(images, training=True)
            mean = outputs["mean"]
            sd = outputs["sd"]
            distribution = tfp.distributions.Normal(mean, sd)

            # Decode images
            latent_space = distribution.sample()
            generated_images = self.decoder(latent_space, training=True)
            reconstruction_loss = self.decoder.compiled_loss(images, generated_images)

            # Loss from discriminator (latent space loss)
            dis = self.discriminator(latent_space, training=True)
            adversarial_loss = self.discriminator.compiled_loss(tf.ones_like(dis), dis)

            # Sum losses
            loss = (
                    # latent_loss * tf.cast(self.latent_shape, tf.float32) +
                    reconstruction_loss * tf.cast(tf.reduce_prod(tf.shape(images)[1:]), tf.float32) +
                    adversarial_loss
            )

        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.encoder.trainable_weights + self.decoder.trainable_weights),
            self.encoder.trainable_weights + self.decoder.trainable_weights)
        )

        # Discriminator
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator(tf.stack([self._latent_prior.sample() for _ in range(images.shape[0])]), training=True)
            discriminated_fake = self.discriminator(latent_space, training=True)
            discriminator_loss = (
                    self.discriminator.compiled_loss(tf.ones_like(discriminated_real), discriminated_real) +
                    self.discriminator.compiled_loss(tf.zeros_like(discriminated_fake), discriminated_fake)
            )

        self.discriminator.optimizer.apply_gradients(zip(
            tape.gradient(discriminator_loss, self.discriminator.trainable_weights),
            self.discriminator.trainable_weights)
        )

        # # Metric (for debugging)
        # self.discriminator.compiled_metrics.update_state(tf.zeros_like(discriminated_fake), discriminated_fake)
        # self.discriminator.compiled_metrics.update_state(tf.ones_like(discriminated_real), discriminated_real)

        return {
            "reconstruction_loss": reconstruction_loss,
            "adversarial_loss": adversarial_loss,
            "loss": loss,
            # "discriminator_accuracy": self.discriminator.metrics[1].result(),
        }

    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)

    def save_all(self, path):
        self.encoder.save(path + "e.h5")
        self.decoder.save(path + "d.h5")
        self.discriminator.save(path + "dis.h5")

    @staticmethod
    def load_all(path: str, latent_prior=None) -> "AAE":
        encoder = tf.keras.models.load_model(path + "e.h5", custom_objects={'Encoder2Normal': Encoder2Normal})
        decoder = tf.keras.models.load_model(path + "d.h5", custom_objects={'Decoder': Decoder})
        discriminator = tf.keras.models.load_model(path + "dis.h5", custom_objects={'Discriminator': Discriminator})
        return AAE(encoder, decoder, discriminator, latent_prior=latent_prior)

    @property
    def string(self):
        return f"aae_l{seq2str(self.latent_shape)};e({self.encoder.string});d({self.decoder.string});di({self.discriminator.string})"
