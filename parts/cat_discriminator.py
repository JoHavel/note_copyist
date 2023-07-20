import tensorflow as tf
from collections.abc import Callable

from utils.my_typing import String, IntSequenceOrInt, IntSequence, seq_or_int_2_seq
from .downsample import Downsample


class CatDiscriminator(tf.keras.Model, String):
    """ Create neural network, that encodes labeled data to True (1) or False (0). """
    _downsample: Downsample
    _base_model_initialized = False  # Hack, so we can set attributes before super().__init__()

    def __init__(
            self,
            input_shape: tuple[IntSequenceOrInt, int],

            hidden_layers: IntSequence = (128,),
            conv_layers: IntSequence = (),

            kernel_sizes: IntSequenceOrInt = 5,
            strides: IntSequenceOrInt = 2,

            hidden_activation: str | Callable = "relu",

            optimizer: tf.keras.optimizers.Optimizer = None,

            name: str = "Discriminator_with_categorical_input",
    ):
        """
            Creates discriminator with `hidden_layers` as units of dense layers (and one dense layer with 1 `units`),
            `conv_layers` as channels of convolutional layers with `strides` and `kernel_sizes` (those are cyclically
            repeated if shorten than `conv_layers`).

            `input_shape[0]` is data input shape, `input_shape[1]` is number of categories.

            Hidden dense and convolutional layers use `hidden_activation` (the last layer uses `sigmoid` as activation
            function).

            Finally, in tf ecosystem this model has `name`, and it is optimized by Optimizer (None = Adam), using
            binary-cross-entropy loss and binary-accuracy metric.
        """
        self._downsample = Downsample(
            input_shape[0], hidden_layers, conv_layers, kernel_sizes, strides, hidden_activation,
            flat=True, input_after_conv_shape=input_shape[1],
        )
        self._downsample.compile()

        inp = [tf.keras.layers.Input(seq_or_int_2_seq(input_shape[0])), tf.keras.layers.Input([input_shape[1]])]
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
