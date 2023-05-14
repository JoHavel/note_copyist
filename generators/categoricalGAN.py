# Based on NPFL114: https://ufal.mff.cuni.cz/courses/npfl114/2122-summer
# FIXME other than shapes [n] (shapes of latent space)

import tensorflow as tf
import tensorflow_probability as tfp


class GAN(tf.keras.Model):
    """ Generative adversarial network (it learns generating images from one-hot labels concatenated with latent space "given" by latent_prior)
        https://ufal.mff.cuni.cz/courses/npfl114/2122-summer#12_deep_generative_models
    """
    def __init__(
            self,
            generator: tf.keras.Model,
            discriminator: tf.keras.Model,
            n_of_categories: int,
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
        self.n_of_categories = n_of_categories

        super().compile()

    def train_step(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        items, labels = data
        categories = tf.one_hot(labels, self.n_of_categories)
        # Generator
        with tf.GradientTape() as tape:
            samples = tf.concat([
                categories,
                self._latent_prior.sample(tf.shape(items)[0], seed=self._seed)[..., self.n_of_categories:]
            ], axis=-1)
            gen = self.generator(samples, training=True)
            dis = self.discriminator((gen, categories), training=True)
            generator_loss = self.discriminator.compiled_loss(tf.ones_like(dis), dis)

        self.generator.optimizer.apply_gradients(
            zip(tape.gradient(generator_loss, self.generator.trainable_weights), self.generator.trainable_weights))

        # Discriminator
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator((items, categories), training=True)
            discriminated_fake = self.discriminator((gen, categories), training=True)
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
