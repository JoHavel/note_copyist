import tensorflow as tf

from .encoder import _body_checks, _conv_downsample, _fully_connected
from .discriminator import _discriminator_head


def _body(
        input_shape: tuple[list[int], int],
        hidden_layers: list[int],
        conv_layers: list[int],
        kernel_size: int,
        stride: int
):  # -> (input, last_layer)
    """ Create the processing part of neural network (input, conv layers, hidden layers) """
    _body_checks(input_shape[0], conv_layers)

    # INPUT
    inp = tf.keras.layers.Input(input_shape[0])
    last_layer = inp

    # LAYERS
    if len(conv_layers) > 0:
        last_layer = _conv_downsample(last_layer, conv_layers, kernel_size, stride, len(input_shape[0]) == 2)

    if len(input_shape[0]) > 1:
        last_layer = tf.keras.layers.Flatten()(last_layer)

    cat_inp = tf.keras.layers.Input([input_shape[1]])
    last_layer = tf.keras.layers.Concatenate()([last_layer, cat_inp])

    if len(hidden_layers) > 0:
        last_layer = _fully_connected(last_layer, hidden_layers)

    return [inp, cat_inp], last_layer


def discriminator(
        input_shape: tuple[list[int], int],
        hidden_layers: list[int] = (128,),
        conv_layers: list[int] = (),
        kernel_size: int = 5,
        stride: int = 2,
        optimizer: tf.keras.optimizers.Optimizer = None,
) -> tf.keras.Model:
    """ Create neural network, that encodes labeled data to True (1) or False (0). """
    inp, last_layer = _body(input_shape, hidden_layers, conv_layers, kernel_size, stride)
    return _discriminator_head(inp, last_layer, optimizer, "Discriminator_with_categorical_input")
