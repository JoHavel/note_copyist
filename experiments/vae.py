from experiments.__init__ import *

from models import encoder, generator
from datasets import mnist
from generators.basicVAE import VAE
from utils.visualizers import gs_img_nd_ls_visualizer

LATENT_SHAPE = [5]
BATCH_SIZE = 50
EPOCHS = 50

network = VAE(
    encoder.encoder_to_normal(mnist.shape, LATENT_SHAPE, hidden_layers=[500, 500]),
    generator.generator(LATENT_SHAPE, mnist.shape, hidden_layers=[500, 500]),
    SEED
)
network.compile(optimizer=tf.optimizers.Adam())


def draw(i, _):
    gs_img_nd_ls_visualizer(network, network.latent_shape, filename=f"out/vae{i}.png")


logs = network.fit(mnist.X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw))
