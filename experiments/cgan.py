from experiments.__init__ import *

from models import cat_discriminator, generator
from datasets import mnist
from generators.categoricalGAN import GAN
from utils.visualizers import cat_gs_img_nd_ls_visualizer

LATENT_SHAPE = [15]
BATCH_SIZE = 50
EPOCHS = 50

network = GAN(
    generator.generator(LATENT_SHAPE, mnist.shape),
    cat_discriminator.discriminator((mnist.shape, mnist.N_OF_CATEGORIES)),
    mnist.N_OF_CATEGORIES,
    SEED
)


def draw(i, _):
    network.save_all("out/models/cgan/cgan" + str(i))
    cat_gs_img_nd_ls_visualizer(network, mnist.N_OF_CATEGORIES, network.latent_shape, filename=f"out/cgan/cgan{i}")


logs = network.fit(
    mnist.X_train,
    mnist.y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
)
