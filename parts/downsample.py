import tensorflow as tf
from collections.abc import Callable

from utils.my_typing import IntSequence, seq_or_int_2_seq, IntSequenceOrInt, String, seq2str
from itertools import cycle


class Downsample(tf.keras.Model, String):
    _intput_shape: IntSequence

    _conv_layers: IntSequence
    _hidden_layers: IntSequence

    _kernel_sizes: IntSequence
    _strides: IntSequence

    _hidden_activation: str | Callable

    _input_after_conv_shape: IntSequence

    _base_model_initialized = False  # Hack, so we can set attributes before super().__init__()

    def __init__(
            self,
            input_shape: IntSequenceOrInt,

            hidden_layers: IntSequence = (128,),
            conv_layers: IntSequence = (),

            kernel_sizes: IntSequenceOrInt = 5,
            strides: IntSequenceOrInt = 2,

            hidden_activation: str | Callable = "relu",

            flat: bool = True,

            input_after_conv_shape: IntSequenceOrInt | None = None,

            name: str = "Downsample",
    ):
        self._input_shape = seq_or_int_2_seq(input_shape)
        self._hidden_layers = hidden_layers
        self._conv_layers = conv_layers
        self._kernel_sizes = seq_or_int_2_seq(kernel_sizes)
        self._strides = seq_or_int_2_seq(strides)
        self._hidden_activation = hidden_activation
        self._input_after_conv_shape = (
            None if input_after_conv_shape is None
            else seq_or_int_2_seq(input_after_conv_shape)
        )

        self._check()

        # INPUT
        inp = tf.keras.layers.Input(self._input_shape)
        last_layer = inp

        # BODY
        last_layer = self._conv(last_layer)
        if len(self._input_shape) > 1 and flat:  # FIXME?
            last_layer = tf.keras.layers.Flatten()(last_layer)

        if input_after_conv_shape is not None:
            inp_after_conv = tf.keras.layers.Input([input_after_conv_shape])
            inp = [inp, inp_after_conv]
            last_layer = tf.keras.layers.Concatenate()([last_layer, inp_after_conv])

        last_layer = self._fully_connected(last_layer)

        # Create tf.Model by funcitonal API
        super().__init__(inputs=inp, outputs=last_layer, name=name)

    def _conv(self, last_layer: tf.Module) -> tf.Module:
        """ Apply convolutional layers TODO Conv1D and Conv3D? """
        if len(self._conv_layers) == 0:
            return last_layer

        if len(self._input_shape) == 2:
            last_layer = last_layer[..., None]

        for filters, kernel_size, strides in \
                zip(self._conv_layers, cycle(self._kernel_sizes), cycle(self._strides)):
            last_layer = tf.keras.layers.Conv2D(
                filters, kernel_size=kernel_size, strides=strides,
                use_bias=False, padding="same",
            )(last_layer)
            last_layer = tf.keras.layers.BatchNormalization()(last_layer)
            last_layer = tf.keras.layers.Activation(activation=self._hidden_activation)(last_layer)

        if len(self._input_shape) == 2:
            last_layer = last_layer[..., 0]

        return last_layer

    def _fully_connected(self, last_layer: tf.Module) -> tf.Module:
        """ Apply fully connected (=dense) layers """
        for layer in self._hidden_layers:
            last_layer = tf.keras.layers.Dense(layer, activation=self._hidden_activation)(last_layer)
        return last_layer

    def _check(self):
        """ Throw exception if the arguments are in conflict or bad. """
        if len(self._input_shape) == 0:
            raise ValueError("Downsample used on scalar data!")

        if len(self._conv_layers) != 0 and len(self._input_shape) == 1:
            raise NotImplemented("Downsample with Conv for 1D data!")

    @property
    def string(self):
        if len(self._conv_layers) == 0:
            return f"hl{seq2str(self._hidden_layers)}"
        return f"hl{seq2str(self._hidden_layers)},cl{seq2str(self._conv_layers)},kern{seq2str(self._kernel_sizes)},strid{seq2str(self._strides)}"
