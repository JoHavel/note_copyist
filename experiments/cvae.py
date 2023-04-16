from experiments.__init__ import *

from models import encoder, generator
from datasets import mnist
from generators.categoricalVAE import VAE
from utils.visualizers import cat_gs_img_nd_ls_visualizer

LATENT_SHAPE = [15]
BATCH_SIZE = 50
EPOCHS = 50

network = VAE(
    encoder.encoder_to_normal(mnist.shape, LATENT_SHAPE, layers=[500, 500]),
    generator.generator(LATENT_SHAPE, mnist.shape, layers=[500, 500]),
    mnist.N_OF_CATEGORIES,
    SEED,
)
network.compile(optimizer=tf.optimizers.Adam())


def draw(i, _):
    cat_gs_img_nd_ls_visualizer(network, mnist.N_OF_CATEGORIES, network.latent_shape, filename=f"out/cvae{i}")


logs = network.fit(
    mnist.X_train,
    mnist.y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
)
