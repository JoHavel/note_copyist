from functools import reduce
from operator import mul

import tensorflow as tf
from collections.abc import Callable

from utils.my_typing import IntSequence, seq_or_int_2_seq, IntSequenceOrInt, String, seq2str
from itertools import cycle


class Upsample(tf.keras.Model, String):
    _intput_shape: IntSequence
    _output_shape: IntSequence

    _conv_layers: IntSequence
    _hidden_layers: IntSequence

    _kernel_sizes: IntSequence
    _strides: IntSequence

    _hidden_activation: str | Callable

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

            name: str = "Downsample",
    ):
        self._input_shape = seq_or_int_2_seq(input_shape)
        self._output_shape = output_shape
        self._hidden_layers = hidden_layers
        self._conv_layers = conv_layers
        self._kernel_sizes = seq_or_int_2_seq(kernel_sizes)
        self._strides = seq_or_int_2_seq(strides)
        self._hidden_activation = hidden_activation

        self._check()

        # INPUT
        inp = tf.keras.layers.Input(self._input_shape)
        last_layer = inp

        # BODY
        last_layer = self._fully_connected(last_layer)
        last_layer = self._conv(last_layer)

        # Create tf.Model by funcitonal API
        super().__init__(inputs=inp, outputs=last_layer, name=name)

    def _conv(self, last_layer: tf.Module) -> tf.Module:
        """ Apply transposed convolutional layers TODO Conv1D and Conv3D? """
        if len(self._conv_layers) == 0:
            return last_layer

        number_of_dims_to_count = len(self._output_shape) - len(self._input_shape)

        if number_of_dims_to_count == 0:
            last_layer = tf.keras.layers.Dense(self._conv_layers[0], activation=self._hidden_activation)(last_layer)
        elif number_of_dims_to_count == 1:  # FIXME?
            last_layer = tf.keras.layers.Dense(self._conv_layers[0], activation=self._hidden_activation)(last_layer[..., None])
        elif number_of_dims_to_count >= 2:
            stride = 1
            for _, strides in zip(self._conv_layers, cycle(self._strides)):
                stride *= strides
            counted_shape = list(self._output_shape[len(self._input_shape)-1:-3]) + [
                self._output_shape[-3] // stride,
                self._output_shape[-2] // stride,
                self._conv_layers[0]
            ]

            last_layer = tf.keras.layers.Dense(reduce(mul, counted_shape), activation="relu")(last_layer)
            last_layer = tf.keras.layers.Reshape(list(self._input_shape[:-1]) + counted_shape)(last_layer)
        else:
            assert False  # Broken state (see self._checks())

        for filters, kernel_size, strides in \
                zip(self._conv_layers[1:], cycle(self._kernel_sizes), cycle(self._strides)):
            last_layer = tf.keras.layers.Conv2DTranspose(
                filters, kernel_size=kernel_size, strides=strides,
                padding="same", use_bias=False
            )(last_layer)
            last_layer = tf.keras.layers.BatchNormalization()(last_layer)
            last_layer = tf.keras.layers.Activation(activation="relu")(last_layer)
        return last_layer

    def _fully_connected(self, last_layer: tf.Module) -> tf.Module:
        """ Apply fully connected (=dense) layers """
        for layer in self._hidden_layers:
            last_layer = tf.keras.layers.Dense(layer, activation=self._hidden_activation)(last_layer)
        return last_layer

    def _check(self):
        """ Throw exception if the arguments are in conflict or bad. """
        if len(self._output_shape) == 0:
            raise ValueError("Upsample used with scalar output!")

        if len(self._conv_layers) != 0 and len(self._output_shape) <= 2:
            raise NotImplemented("Upample with Conv for 1D output!")

        if len(self._input_shape) > len(self._output_shape):
            raise ValueError("Upsample used with outputD smaller than inputD!")

    @property
    def string(self):
        if len(self._conv_layers) == 0:
            return f"hl{seq2str(self._hidden_layers)}"
        return f"hl{seq2str(self._hidden_layers)},cl{seq2str(self._conv_layers)},kern{seq2str(self._kernel_sizes)},strid{seq2str(self._strides)}"
