from experiments.__init__ import *

from models import encoder, generator, discriminator
from datasets import mnist
from generators.basicAAE import AAE
from utils.visualizers import gs_img_nd_ls_visualizer

LATENT_SHAPE = [5]
BATCH_SIZE = 50
EPOCHS = 50

network = AAE(
    encoder.encoder_to_normal(mnist.shape, LATENT_SHAPE, hidden_layers=[500, 500]),
    generator.generator(LATENT_SHAPE, mnist.shape, layers=[500, 500]),
    discriminator.discriminator(LATENT_SHAPE),
    SEED,
)
network.compile(optimizer=tf.optimizers.Adam())


def draw(i, _):
    gs_img_nd_ls_visualizer(network, network.latent_shape, filename=f"out/aae{i}.png")


logs = network.fit(
    mnist.X_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
)
