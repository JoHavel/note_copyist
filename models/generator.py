from functools import reduce
from operator import mul

import tensorflow as tf


def generator(
        input_shape: list[int],
        output_shape: list[int],
        output_activation="sigmoid",
        layers: list[int] = [128],
        conv_layers: list[int] = [],
        kernel_size: int = 4,
        stride: int = 2,
        optimizer: tf.keras.optimizers.Optimizer = None,
) -> tf.keras.Model:
    if len(conv_layers) != 0 and len(output_shape) < 2:
        raise ValueError("Conv for 1D or scalar data!")

    inp = tf.keras.layers.Input(input_shape)
    last_layer = inp

    for layer in layers:
        last_layer = tf.keras.layers.Dense(layer, activation="relu")(last_layer)

    if len(input_shape) == 1:
        counted_shape = [
            output_shape[0] // stride ** (len(conv_layers)),
            output_shape[1] // stride ** (len(conv_layers)),
            conv_layers[0] if len(conv_layers) > 0 else 1
        ]

        if len(conv_layers) > 0:
            last_layer = tf.keras.layers.Dense(reduce(mul, counted_shape), activation="relu")(last_layer)
        else:
            last_layer = tf.keras.layers.Dense(reduce(mul, counted_shape), activation=output_activation)(last_layer)

        last_layer = tf.keras.layers.Reshape(counted_shape)(last_layer)

    if len(conv_layers) > 0:
        if len(output_shape) == 2:
            last_layer = last_layer[..., None]

        for cl in conv_layers[1:]:
            last_layer = tf.keras.layers.Conv2DTranspose(cl, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(last_layer)
            last_layer = tf.keras.layers.BatchNormalization()(last_layer)
            last_layer = tf.keras.layers.Activation(activation="relu")(last_layer)
        last_layer = tf.keras.layers.Conv2DTranspose(1, kernel_size=kernel_size, strides=stride, activation=output_activation, padding="same")(last_layer)

        if len(output_shape) == 2:
            last_layer = last_layer[..., 0]

    model = tf.keras.Model(inputs=inp, outputs=last_layer, name="Generator")

    model.compile(
        optimizer=optimizer if optimizer is not None else tf.keras.optimizers.Adam(),
        loss=tf.losses.BinaryCrossentropy(),
    )

    return model
