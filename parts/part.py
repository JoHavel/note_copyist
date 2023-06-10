import tensorflow as tf


class Part(tf.keras.Model):
    def __init__(self, string: str = "", **kwargs):
        super().__init__(**kwargs)
        self.string = string

    # def __str__(self):
    #     return self.string
