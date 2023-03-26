# Based on NPFL114: https://ufal.mff.cuni.cz/courses/npfl114/2122-summer

import tensorflow as tf
import tensorflow_probability as tfp


class VAE(tf.keras.Model):
    def __init__(
            self,
            encoder: tf.keras.Model,
            decoder: tf.keras.Model,
            seed: int = 42,
            latent_prior=None,
    ) -> None:
        super().__init__()

        self._seed = seed
        self.latent_shape = decoder.inputs[0].shape[1:]

        if latent_prior is None:
            self._latent_prior = tfp.distributions.Normal(tf.zeros(self.latent_shape), tf.ones(self.latent_shape))
        else:
            self._latent_prior = latent_prior

        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, images: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            # Encode to latent space distribution
            outputs = self.encoder(images, training=True)
            mean = outputs["mean"]
            sd = outputs["sd"]
            distribution = tfp.distributions.Normal(mean, sd)

            # Decode images
            latent_space = distribution.sample(seed=self._seed)
            generated_images = self.decoder(latent_space, training=True)
            reconstruction_loss = self.decoder.compiled_loss(images, generated_images)

            # Loss from "difference" from prior
            latent_loss = tf.reduce_mean(
                distribution.kl_divergence(self._latent_prior)
            )

            # Sum losses
            loss = (
                    latent_loss * tf.cast(self.latent_shape, tf.float32) +
                    reconstruction_loss * tf.cast(tf.reduce_prod(tf.shape(images)[1:-1]), tf.float32)
            )

        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.encoder.trainable_weights + self.decoder.trainable_weights),
            self.encoder.trainable_weights + self.decoder.trainable_weights)
        )

        return {"reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss, "loss": loss}

    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)
