import tensorflow as tf
import tensorflow_probability as tfp
import cv2 as cv
import numpy as np


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


class AdaptiveBinarize(tf.keras.Model):
    """
        Binarize image outputted by model with adaptive threshold from openCV
        (threshold is computed from neighbourhood of pixel).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, *args, **kwargs):
        int_im = tf.cast(self.model(*args, **kwargs)[..., 0].numpy()*255, tf.uint8).numpy()
        binarized_im = np.zeros_like(int_im)
        for i in range(len(int_im)):
            binarized_im[i] = cv.adaptiveThreshold(int_im[i], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, -2)
            # binarized_im[i] = cv.threshold(int_im[i], 0, 255, cv.THRESH_OTSU)[1]
        return binarized_im/255
