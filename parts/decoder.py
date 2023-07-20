import tensorflow as tf
from collections.abc import Callable
from functools import reduce
from operator import mul

from parts.upsample import Upsample
from utils.my_typing import IntSequenceOrInt, IntSequence, String, seq_or_int_2_seq

_MAGIC_LIMIT = 4096
""" Empiric chosen constant. In greater dimensional space, we should not use binary-cross-entropy """


class Decoder(tf.keras.Model, String):
    """ Neural network, that decodes latent data to data """
    _upsample: Upsample
    _base_model_initialized = False  # Hack, so we can set attributes before super().__init__()

    def __init__(
            self,
            input_shape: IntSequenceOrInt,
            output_shape: IntSequence,

            hidden_layers: IntSequence = (128,),
            conv_layers: IntSequence = (),

            kernel_sizes: IntSequenceOrInt = 5,
            strides: IntSequenceOrInt = 2,

            hidden_activation: str | Callable = "relu",

            name: str = "Decoder",

            optimizer: tf.keras.optimizers.Optimizer = None,
            loss: tf.keras.losses.Loss = None,
            output_activation: str | Callable = "sigmoid"
    ):
        """
            Creates decoder with `hidden_layers` as units of dense layers (and one dense layer with output_shape[-1]
            `units`), `conv_layers` as channels of convolutional layers with `strides` and `kernel_sizes` (those are
            cyclically repeated if shorten than `conv_layers`).

            It takes point from `input_shape` space and returns data in `output_shape` space.

            Hidden dense and convolutional layers use `hidden_activation` (the last layer uses output_activation).

            Finally, in tf ecosystem this model has `name`, and it is optimized by Optimizer (None = Adam), using
            binary-cross-entropy or MSE loss (depending on `ouput_shape`>`_MAGIC_LIMIT`).
        """
        self._upsample = Upsample(
            input_shape,
            output_shape if len(output_shape) != 2 else list(output_shape) + [1],
            hidden_layers, conv_layers, kernel_sizes, strides, hidden_activation, name
        )
        self._upsample.compile()

        input_shape = seq_or_int_2_seq(input_shape)

        inp = tf.keras.layers.Input(input_shape)
        last_layer = self._upsample(inp)
        if len(conv_layers) == 0:
            last_layer = tf.keras.layers.Dense(reduce(mul, output_shape[len(input_shape) - 1:]), activation=output_activation)(last_layer)
            last_layer = tf.keras.layers.Reshape(output_shape)(last_layer)
        else:
            channels = output_shape[-1] if len(output_shape) != 2 else 1
            last_layer = tf.keras.layers.Conv2DTranspose(
                channels,
                kernel_size=kernel_sizes if isinstance(kernel_sizes, int) else kernel_sizes[(len(conv_layers) - 1) % len(kernel_sizes)],
                strides=strides if isinstance(strides, int) else strides[(len(conv_layers) - 1) % len(strides)],
                activation=output_activation, padding="same"
            )(last_layer)

            if len(output_shape) == 2:
                last_layer = last_layer[..., 0]

        super().__init__(inputs=inp, outputs=last_layer, name=name)
        if loss is None:
            if reduce(mul, output_shape) < _MAGIC_LIMIT:
                loss = tf.losses.BinaryCrossentropy()
            else:
                loss = tf.losses.MeanSquaredError()

        self.compile(
            optimizer=optimizer if optimizer is not None else tf.keras.optimizers.Adam(),
            loss=loss,
        )

    @property
    def string(self):
        return self._upsample.string
