from functools import reduce
from operator import mul

import tensorflow as tf
from .encoder import _fully_connected


def _body_checks(input_shape: list[int], output_shape: list[int], conv_layers: list[int]):
    """ Throw exception if the arguments are in conflict. """
    if len(conv_layers) != 0 and len(output_shape) < 2:
        raise ValueError("Conv for 1D or scalar data!")

    # TODO check compatibility of input_shape and output_shape if len(input_shape) > 1 and len(output_shape) > 1


def _conv_upsample(last_layer, conv_layers: list[int], kernel_size: int, stride: int, add_dim: bool, output_activation, output_shape: list[int],):  # -> last_layer
    """ Apply transposed convolutional layers """
    if len(conv_layers) == 0:
        return last_layer

    if add_dim:
        last_layer = last_layer[..., None]

    for cl in conv_layers[1:]:
        last_layer = tf.keras.layers.Conv2DTranspose(cl, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(last_layer)
        last_layer = tf.keras.layers.BatchNormalization()(last_layer)
        last_layer = tf.keras.layers.Activation(activation="relu")(last_layer)
    last_layer = tf.keras.layers.Conv2DTranspose(1 if add_dim else output_shape[-1], kernel_size=kernel_size, strides=stride, activation=output_activation, padding="same")(last_layer)

    if add_dim:
        last_layer = last_layer[..., 0]

    return last_layer


def _network(
        input_shape: list[int],
        output_shape: list[int],
        hidden_layers: list[int],
        conv_layers: list[int],
        kernel_size: int,
        stride: int,
        output_activation,
):  # -> (input, last_layer)
    """ Create a neural network (input, hidden layers, transposed conv layers) for generator """
    _body_checks(input_shape, output_shape, conv_layers)

    # INPUT
    inp = tf.keras.layers.Input(input_shape)
    last_layer = inp

    # LAYERS
    if len(input_shape) > 1 and len(output_shape) == 1:
        last_layer = tf.keras.layers.Flatten()(last_layer)

    last_layer = _fully_connected(last_layer, hidden_layers)

    # Prepare for convolutional layers
    if len(input_shape) == 1 and len(conv_layers) > 0:
        counted_shape = output_shape[:-2] + [
            output_shape[-2] // stride ** (len(conv_layers)),
            output_shape[-1] // stride ** (len(conv_layers)),
            conv_layers[0]
        ]

        last_layer = tf.keras.layers.Dense(reduce(mul, counted_shape), activation="relu")(last_layer)
        last_layer = tf.keras.layers.Reshape(counted_shape)(last_layer)

    # If dense layers are last...
    if len(conv_layers) == 0:
        if len(input_shape) == 1:
            last_layer = tf.keras.layers.Dense(reduce(mul, output_shape), activation=output_activation)(last_layer)
            if len(output_shape) > 1:
                last_layer = tf.keras.layers.Reshape(output_shape)(last_layer)
        else:
            last_layer = tf.keras.layers.Dense(output_shape[-1], activation=output_activation)(last_layer)

    last_layer = _conv_upsample(last_layer, conv_layers, kernel_size, stride, len(output_shape) == 2, output_activation, output_shape)

    return inp, last_layer


def generator(
        input_shape: list[int],
        output_shape: list[int],
        output_activation="sigmoid",
        hidden_layers: list[int] = (128,),
        conv_layers: list[int] = (),
        kernel_size: int = 4,
        stride: int = 2,
        optimizer: tf.keras.optimizers.Optimizer = None,
) -> tf.keras.Model:
    """ Create neural network, that decodes latent data to data """
    inp, last_layer =\
        _network(input_shape, output_shape, hidden_layers, conv_layers, kernel_size, stride, output_activation)

    model = tf.keras.Model(inputs=inp, outputs=last_layer, name="Generator")

    model.compile(
        optimizer=optimizer if optimizer is not None else tf.keras.optimizers.Adam(),
        loss=tf.losses.BinaryCrossentropy(),
    )

    return model
