# Based on basicAAE and https://arxiv.org/pdf/1511.05644.pdf.
# FIXME other than shapes [n] (shapes of latent space)

import tensorflow as tf
import tensorflow_probability as tfp


class AAE(tf.keras.Model):
    """ Adversarial auto encoder (it learns generating images from one-hot labels concatenated with latent space "given" by latent_prior)
        https://medium.com/vitrox-publication/adversarial-auto-encoder-aae-a3fc86f71758,
        https://arxiv.org/pdf/1511.05644.pdf.
    """
    def __init__(
            self,
            encoder: tf.keras.Model,
            decoder: tf.keras.Model,
            discriminator: tf.keras.Model,
            n_of_categories: int,
            seed: int = 42,
            latent_prior=None,
    ) -> None:
        super().__init__()

        self._seed = seed
        self.latent_shape = encoder.outputs[0].shape[1:]

        if latent_prior is None:
            self._latent_prior = tfp.distributions.Normal(tf.zeros(self.latent_shape), tf.ones(self.latent_shape))
        else:
            self._latent_prior = latent_prior

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.n_of_categories = n_of_categories

    def train_step(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        images, labels = data
        # Generator
        with tf.GradientTape() as tape:
            # Encode to latent space distribution
            outputs = self.encoder(images, training=True)
            mean = outputs["mean"]
            sd = outputs["sd"]
            distribution = tfp.distributions.Normal(mean, sd)

            # Decode images
            latent_space = distribution.sample(seed=self._seed)
            generated_images = self.decoder(
                tf.concat([tf.one_hot(labels, self.n_of_categories), latent_space], axis=-1),
                training=True
            )
            reconstruction_loss = self.decoder.compiled_loss(images, generated_images)

            # Loss from discriminator (latent space loss)
            dis = self.discriminator(latent_space, training=True)
            adversarial_loss = self.discriminator.compiled_loss(tf.ones_like(dis), dis)

            # Sum losses
            loss = (
                    # latent_loss * tf.cast(self.latent_shape, tf.float32) +
                    reconstruction_loss * tf.cast(tf.reduce_prod(tf.shape(images)[1:-1]), tf.float32) +
                    adversarial_loss
            )

        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.encoder.trainable_weights + self.decoder.trainable_weights),
            self.encoder.trainable_weights + self.decoder.trainable_weights)
        )

        # Discriminator
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator(tf.stack([self._latent_prior.sample() for _ in range(images.shape[0])]), training=True)  # FIXME seed=self._seed?
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
