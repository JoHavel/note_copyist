from collections.abc import Callable

import tensorflow as tf

from parts.downsample import Downsample
from utils.my_typing import IntSequenceOrInt, IntSequence, String, seq_or_int_2_seq


class Encoder2Normal(tf.keras.Model, String):
    """
        Neural network, that encodes data to mean and standard deviation of multidimensional normal distribution.
    """
    # TODO: Check if output_shape[:-1] matches output of conv. layers [:-1]

    _downsample: Downsample
    _base_model_initialized = False  # Hack, so we can set attributes before super().__init__()

    def __init__(
            self,
            input_shape: IntSequenceOrInt,
            output_shape: IntSequenceOrInt,

            hidden_layers: IntSequence = (128,),
            conv_layers: IntSequence = (),

            kernel_sizes: IntSequenceOrInt = 5,
            strides: IntSequenceOrInt = 2,

            hidden_activation: str | Callable = "relu",

            flat: bool = True
    ):
        self._downsample = Downsample(
            input_shape, hidden_layers, conv_layers, kernel_sizes, strides, hidden_activation, flat,
            input_after_conv_shape=None,
        )
        self._downsample.compile()

        input_shape = seq_or_int_2_seq(input_shape)
        output_shape = seq_or_int_2_seq(output_shape)

        inp = tf.keras.layers.Input(input_shape)
        last_layer = self._downsample(inp)
        mean = tf.keras.layers.Dense(output_shape[-1])(last_layer)
        sd = tf.keras.layers.Dense(output_shape[-1], activation="exponential")(last_layer)
        super().__init__(inputs=inp, outputs={"mean": mean, "sd": sd}, name="Encoder2Normal")
        self.compile(loss=tf.losses.MSE)

    @property
    def string(self):
        return self._downsample.string
