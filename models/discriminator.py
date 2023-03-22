import tensorflow as tf


def discriminator(
        input_shape: list[int],
        layers: list[int] = [128],
        conv_layers: list[int] = [],
        kernel_size: int = 5,
        stride: int = 2,
        optimizer: tf.keras.optimizers.Optimizer = None,
) -> tf.keras.Model:
    if len(conv_layers) != 0 and len(input_shape) < 2:
        raise ValueError("Conv for 1D or scalar data!")

    inp = tf.keras.layers.Input(input_shape)
    last_layer = inp

    if len(input_shape) > 1:
        if len(conv_layers) > 0 and len(input_shape) == 2:
            last_layer = last_layer[..., None]

        for cl in conv_layers:
            last_layer = tf.keras.layers.Conv2D(cl, kernel_size=kernel_size, strides=stride, use_bias=False)(last_layer)
            last_layer = tf.keras.layers.BatchNormalization()(last_layer)
            last_layer = tf.keras.layers.Activation(activation="relu")(last_layer)

        last_layer = tf.keras.layers.Flatten()(last_layer)

    for layer in layers:
        last_layer = tf.keras.layers.Dense(layer, activation="relu")(last_layer)

    last_layer = tf.keras.layers.Dense(1, activation="sigmoid")(last_layer)[0]

    model = tf.keras.Model(inputs=inp, outputs=last_layer, name="Discriminator")

    model.compile(
        optimizer=optimizer if optimizer is not None else tf.keras.optimizers.Adam(),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=tf.metrics.BinaryAccuracy(),
    )

    return model
