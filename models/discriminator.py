import tensorflow as tf

from .encoder import _body


def _discriminator_head(inp, last_layer, optimizer: tf.keras.optimizers.Optimizer, name: str) -> tf.keras.Model:
    """ Add the outputting part of discriminator to _body and compile it with Binary... """
    last_layer = tf.keras.layers.Dense(1, activation="sigmoid")(last_layer)[0]

    model = tf.keras.Model(inputs=inp, outputs=last_layer, name=name)

    model.compile(
        optimizer=optimizer if optimizer is not None else tf.keras.optimizers.Adam(),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=tf.metrics.BinaryAccuracy(),
    )

    return model


def discriminator(
        input_shape: list[int],
        hidden_layers: list[int] = (128,),
        conv_layers: list[int] = (),
        kernel_size: int = 5,
        stride: int = 2,
        optimizer: tf.keras.optimizers.Optimizer = None,
) -> tf.keras.Model:
    """ Create neural network, that encodes data to True (1) or False (0). """
    inp, last_layer = _body(input_shape, [1], hidden_layers, conv_layers, kernel_size, stride)
    return _discriminator_head(inp, last_layer, optimizer, "Discriminator")
