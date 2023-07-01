import tensorflow as tf


class Part(tf.keras.Model):
    def __init__(self, inputs, outputs, name=None, trainable=True, **kwargs):
        if "string" in kwargs:
            string = kwargs["string"]
            del(kwargs["string"])
        else:
            string = ""
        super().__init__(inputs=inputs, outputs=outputs, name=name, trainable=trainable, **kwargs)
        self.string = string

    # def __str__(self):
    #     return self.string
