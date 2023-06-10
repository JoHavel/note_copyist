import tensorflow as tf


def _body_checks(input_shape: list[int], conv_layers: list[int]):
    """ Throw exception if the arguments are in conflict. """
    if len(conv_layers) != 0 and len(input_shape) < 2:
        raise ValueError("Conv for 1D or scalar data!")


def _conv_downsample(last_layer, conv_layers: list[int], kernel_size: int, stride: int, add_dim: bool):  # -> last_layer
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


def _fully_connected(last_layer, layers: list[int]):  # -> last_layer
    """ Apply fully connected (=dense) layers """
    for layer in layers:
        last_layer = tf.keras.layers.Dense(layer, activation="relu")(last_layer)
    return last_layer


def _body(
        input_shape: list[int],
        output_shape: list[int],
        hidden_layers: list[int],
        conv_layers: list[int],
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


def _normal_dist_head(inp, last_layer, output_shape: list[int], name: str) -> tf.keras.Model:
    """ Add the outputting part of encoder_to_normal to _body """
    mean = tf.keras.layers.Dense(output_shape[-1])(last_layer)
    sd = tf.keras.layers.Dense(output_shape[-1], activation="exponential")(last_layer)
    return tf.keras.Model(inputs=inp, outputs={"mean": mean, "sd": sd}, name=name)


def encoder_to_normal(
        input_shape: list[int],
        output_shape: list[int],
        hidden_layers: list[int] = (128,),
        conv_layers: list[int] = (),
        kernel_size: int = 5,
        stride: int = 2
) -> tf.keras.Model:
    """
        Create neural network, that encodes data to mean and standard deviation of multidimensional normal distribution.
    """
    # TODO: Check if output_shape[:-1] matches output of conv. layers [:-1]
    inp, last_layer = _body(input_shape, output_shape, hidden_layers, conv_layers, kernel_size, stride)
    model = _normal_dist_head(inp, last_layer, output_shape, "Encoder_to_normal_dist")
    model.compile(loss=tf.losses.MSE)
    return model
