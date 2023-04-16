from experiments.__init__ import *

from models import discriminator, generator
from datasets import mnist
from generators.basicGAN import GAN
from utils.visualizers import gs_img_nd_ls_visualizer

LATENT_SHAPE = [100]
BATCH_SIZE = 50
EPOCHS = 50

network = GAN(generator.generator(LATENT_SHAPE, mnist.shape), discriminator.discriminator(mnist.shape), SEED)


def draw(i, _):
    gs_img_nd_ls_visualizer(network, network.latent_shape, filename=f"out/gan{i}.png")


logs = network.fit(
    mnist.X_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
)
