import tensorflow as tf

class Binarize(tf.keras.Model):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        return tf.where(self.model(*args, **kwargs) > self.threshold, 1.0, 0.0)