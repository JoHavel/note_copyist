import tensorflow as tf
from collections.abc import Callable

from utils.my_typing import IntSequenceOrInt, IntSequence, String, seq_or_int_2_seq
from .downsample import Downsample


class Discriminator(tf.keras.Model, String):  # FIXME typing (Discriminator is not CatDiscriminator)
    """ Neural network, that encodes data to True (1) or False (0). """
    _downsample: Downsample
    _base_model_initialized = False  # Hack, so we can set attributes before super().__init__()

    def __init__(
            self,
            input_shape: IntSequenceOrInt,

            hidden_layers: IntSequence = (128,),
            conv_layers: IntSequence = (),

            kernel_sizes: IntSequenceOrInt = 5,
            strides: IntSequenceOrInt = 2,

            hidden_activation: str | Callable = "relu",

            optimizer: tf.keras.optimizers.Optimizer = None,

            name: str = "Discriminator",
    ):
        """
            Creates discriminator with `hidden_layers` as units of dense layers (and one dense layer with 1 `units`),
            `conv_layers` as channels of convolutional layers with `strides` and `kernel_sizes` (those are cyclically
            repeated if shorten than `conv_layers`).

            Hidden dense and convolutional layers use `hidden_activation` (the last layer uses `sigmoid` as activation
            function).

            Finally, in tf ecosystem this model has `name`, and it is optimized by Optimizer (None = Adam), using
            binary-cross-entropy loss and binary-accuracy metric.
        """
        self._downsample = Downsample(
            input_shape, hidden_layers, conv_layers, kernel_sizes, strides, hidden_activation,
            flat=True,
        )
        self._downsample.compile()

        input_shape = seq_or_int_2_seq(input_shape)

        inp = tf.keras.layers.Input(input_shape)
        last_layer = self._downsample(inp)
        last_layer = tf.keras.layers.Dense(1, activation="sigmoid")(last_layer)[..., 0]

        super().__init__(inputs=inp, outputs=last_layer, name=name)
        self.compile(
            optimizer=optimizer if optimizer is not None else tf.keras.optimizers.Adam(),
            loss=tf.losses.BinaryCrossentropy(),
            metrics=tf.metrics.BinaryAccuracy(),
        )

    @property
    def string(self):
        return self._downsample.string
