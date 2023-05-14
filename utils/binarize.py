import tensorflow as tf
import tensorflow_probability as tfp


class Binarize(tf.keras.Model):
    """ Binarize image outputted by model with fixed threshold (all above -> 1.0, all below -> 0.0) """
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        return tf.where(self.model(*args, **kwargs) > self.threshold, 1.0, 0.0)


class ProbBinarize(tf.keras.Model):
    """ Binarize image outputted by model as if every value was a parameter of the Bernoulli distribution. """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, *args, **kwargs):
        return tfp.distributions.Bernoulli(probs=self.model(*args, **kwargs)).sample()
