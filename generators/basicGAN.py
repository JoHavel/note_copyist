# Based on NPFL114: https://ufal.mff.cuni.cz/courses/npfl114/2122-summer

import tensorflow as tf
import tensorflow_probability as tfp

from parts.discriminator import Dis
from parts.generator import Gen


class GAN(tf.keras.Model):
    """ Generative adversarial network (it learns generating images from latent space "given" by latent_prior)
        https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#12_deep_generative_models
    """
    def __init__(
            self,
            generator: Gen,
            discriminator: Dis,
            seed: float = 42,
            latent_prior=None,
    ) -> None:
        """

        :param generator: input in the shape of latent space, output in the shape of generated items
        :param discriminator: input in the shape of generator's output, outputs one number (in range [0, 1])
        """
        super().__init__()

        self._seed = seed
        self.latent_shape = generator.inputs[0].shape[1:]

        if latent_prior is None:
            self._latent_prior = tfp.distributions.Normal(tf.zeros(self.latent_shape), tf.ones(self.latent_shape))
        else:
            self._latent_prior = latent_prior

        self.generator = generator
        self.discriminator = discriminator

        super().compile()

    def train_step(self, items: tf.Tensor) -> dict[str, tf.Tensor]:
        # Generator
        with tf.GradientTape() as tape:
            samples = self._latent_prior.sample(tf.shape(items)[0], seed=self._seed)
            gen = self.generator(samples, training=True)
            dis = self.discriminator(gen, training=True)
            generator_loss = self.discriminator.compiled_loss(tf.ones_like(dis), dis)

        self.generator.optimizer.apply_gradients(
            zip(tape.gradient(generator_loss, self.generator.trainable_weights), self.generator.trainable_weights))

        # Discriminator
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator(items, training=True)
            discriminated_fake = self.discriminator(gen, training=True)
            discriminator_loss = (
                    self.discriminator.compiled_loss(tf.ones_like(discriminated_real), discriminated_real) +
                    self.discriminator.compiled_loss(tf.zeros_like(discriminated_fake), discriminated_fake)
            )

        self.discriminator.optimizer.apply_gradients(zip(
            tape.gradient(discriminator_loss, self.discriminator.trainable_weights),
            self.discriminator.trainable_weights)
        )

        # Metric
        self.discriminator.compiled_metrics.update_state(tf.zeros_like(discriminated_fake), discriminated_fake)
        self.discriminator.compiled_metrics.update_state(tf.ones_like(discriminated_real), discriminated_real)

        return {
            "discriminator_loss": discriminator_loss,
            "discriminator_accuracy": self.discriminator.metrics[1].result(),
            "generator_loss": generator_loss,
        }

    def call(self, inputs, **kwargs):
        return self.generator(inputs, **kwargs)

    def save_all(self, path):
        self.generator.save(path + "g.h5")
        self.discriminator.save(path + "d.h5")

    @staticmethod
    def load_all(path: str, latent_prior=None):  # -> GAN
        generator = tf.keras.models.load_model(path + "g.h5")
        discriminator = tf.keras.models.load_model(path + "d.h5")
        return GAN(generator, discriminator, latent_prior=latent_prior)
