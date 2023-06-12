from typing import TypeAlias

import tensorflow as tf

from parts.part import Part


def _body_checks(input_shape: list[int] | tuple[int, ...], conv_layers: list[int] | tuple[int, ...]):
    """ Throw exception if the arguments are in conflict. """
    if len(conv_layers) != 0 and len(input_shape) < 2:
        raise ValueError("Conv for 1D or scalar data!")


def _conv_downsample(last_layer, conv_layers: list[int] | tuple[int, ...], kernel_size: int, stride: int, add_dim: bool):  # -> last_layer
    """ Apply convolutional layers """
    if len(conv_layers) == 0:
        return last_layer

    if add_dim:
        last_layer = last_layer[..., None]

    for cl in conv_layers:
        last_layer = tf.keras.layers.Conv2D(cl, kernel_size=kernel_size, strides=stride, use_bias=False, padding="same")(last_layer)
        last_layer = tf.keras.layers.BatchNormalization()(last_layer)
        last_layer = tf.keras.layers.Activation(activation="relu")(last_layer)

    if add_dim:
        last_layer = last_layer[..., 0]

    return last_layer


def _fully_connected(last_layer, layers: list[int] | tuple[int, ...]):  # -> last_layer
    """ Apply fully connected (=dense) layers """
    for layer in layers:
        last_layer = tf.keras.layers.Dense(layer, activation="relu")(last_layer)
    return last_layer


def _body(
        input_shape: list[int] | tuple[int, ...],
        output_shape: list[int] | tuple[int, ...],
        hidden_layers: list[int] | tuple[int, ...],
        conv_layers: list[int] | tuple[int, ...],
        kernel_size: int,
        stride: int
):  # -> (input, last_layer)
    """ Create the processing part of neural network (input, conv layers, hidden layers) """
    _body_checks(input_shape, conv_layers)

    # INPUT
    inp = tf.keras.layers.Input(input_shape)
    last_layer = inp

    # LAYERS
    last_layer = _conv_downsample(last_layer, conv_layers, kernel_size, stride, len(input_shape) == 2)

    if len(input_shape) > 1 and len(output_shape) == 1:
        last_layer = tf.keras.layers.Flatten()(last_layer)

    last_layer = _fully_connected(last_layer, hidden_layers)

    return inp, last_layer


Enc: TypeAlias = Part


def _normal_dist_head(inp, last_layer, output_shape: list[int] | tuple[int, ...], name: str) -> Enc:
    """ Add the outputting part of encoder_to_normal to _body """
    mean = tf.keras.layers.Dense(output_shape[-1])(last_layer)
    sd = tf.keras.layers.Dense(output_shape[-1], activation="exponential")(last_layer)
    return Enc(inputs=inp, outputs={"mean": mean, "sd": sd}, name=name)


def encoder_to_normal(
        input_shape: list[int] | tuple[int, ...],
        output_shape: list[int] | tuple[int, ...],
        hidden_layers: list[int] | tuple[int, ...] = (128,),
        conv_layers: list[int] | tuple[int, ...] = (),
        kernel_size: int = 5,
        stride: int = 2
) -> Enc:
    """
        Create neural network, that encodes data to mean and standard deviation of multidimensional normal distribution.
    """
    # TODO: Check if output_shape[:-1] matches output of conv. layers [:-1]
    inp, last_layer = _body(input_shape, output_shape, hidden_layers, conv_layers, kernel_size, stride)
    model = _normal_dist_head(inp, last_layer, output_shape, "Encoder_to_normal_dist")
    model.compile(loss=tf.losses.MSE)
    model.string = f"{hidden_layers},{conv_layers},{kernel_size},{stride}"
    return model
