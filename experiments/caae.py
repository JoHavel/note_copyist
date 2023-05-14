from experiments.__init__ import *

from models import encoder, generator, discriminator
from datasets import mnist
from generators.categoricalAAE import AAE
from utils.visualizers import cat_gs_img_2d_ls_visualizer

LATENT_SHAPE = [2]
BATCH_SIZE = 50
EPOCHS = 50

network = AAE(
    encoder.encoder_to_normal(mnist.shape, LATENT_SHAPE, hidden_layers=[500, 500]),
    generator.generator([LATENT_SHAPE[0] + mnist.N_OF_CATEGORIES], mnist.shape, hidden_layers=[500, 500]),
    discriminator.discriminator(LATENT_SHAPE),
    mnist.N_OF_CATEGORIES,
    SEED,
)
network.compile(optimizer=tf.optimizers.Adam())


def draw(i, _):
    cat_gs_img_2d_ls_visualizer(
        network,
        mnist.N_OF_CATEGORIES,
        filename=f"out/caae{i}",
    )


logs = network.fit(
    mnist.X_train,
    mnist.y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
)
