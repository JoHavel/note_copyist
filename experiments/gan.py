from experiments.__init__ import *

from models import discriminator, generator
from datasets import mnist
from generators.basicGAN import GAN
from validation.visualizers import gs_img_nd_ls_visualizer

network = GAN(generator.generator(LATENT_SHAPE, mnist.shape), discriminator.discriminator(mnist.shape), SEED)


def draw(i, _):
    gs_img_nd_ls_visualizer(network, network.latent_shape, filename=f"out/gan{i}.png")


logs = network.fit(
    mnist.X_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
)
