import os.path
import unittest
import tempfile

from parts.encoder import Encoder2Normal
from parts.decoder import Decoder
from parts.discriminator import Discriminator
from parts.cat_discriminator import CatDiscriminator
from utils.tests import test_string

from .basicGAN import GAN
from .basicVAE import VAE
from .basicAAE import AAE
from .categoricalGAN import GAN as CGAN
from .categoricalVAE import VAE as CVAE
from .categoricalAAE import AAE as CAAE


class TestSavingAndLoadingGenerators(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_gan(self):
        filename = os.path.join(self.temp_dir.name, "gan")
        gan = GAN(Decoder(1, [10, 10]), Discriminator([10, 10]))
        gan.save_all(filename)
        GAN.load_all(filename)

    def test_vae(self):
        filename = os.path.join(self.temp_dir.name, "vae")
        vae = VAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]))
        vae.save_all(filename)
        VAE.load_all(filename)

    def test_aae(self):
        filename = os.path.join(self.temp_dir.name, "aae")
        aae = AAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]), Discriminator(1))
        aae.save_all(filename)
        AAE.load_all(filename)

    def test_cgan(self):
        filename = os.path.join(self.temp_dir.name, "gan")
        gan = CGAN(Decoder(1, [10, 10]), CatDiscriminator(([10, 10], 1)), 1)
        gan.save_all(filename)
        CGAN.load_all(filename)

    def test_cvae(self):
        filename = os.path.join(self.temp_dir.name, "cvae")
        vae = CVAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]), 1)
        vae.save_all(filename)
        CVAE.load_all(filename)

    def test_caae(self):
        filename = os.path.join(self.temp_dir.name, "aae")
        aae = CAAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]), Discriminator(1), 1)
        aae.save_all(filename)
        CAAE.load_all(filename)


class TestString(unittest.TestCase):
    def test_gan(self):
        gan = GAN(Decoder(1, [10, 10]), Discriminator([10, 10]))
        test_string(self, gan)

    def test_vae(self):
        vae = VAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]))
        test_string(self, vae)

    def test_aae(self):
        aae = AAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]), Discriminator(1))
        test_string(self, aae)

    def test_cgan(self):
        gan = CGAN(Decoder(1, [10, 10]), CatDiscriminator(([10, 10], 1)), 1)
        test_string(self, gan)

    def test_cvae(self):
        vae = CVAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]), 1)
        test_string(self, vae)

    def test_caae(self):
        aae = CAAE(Encoder2Normal([10, 10], 1), Decoder(1, [10, 10]), Discriminator(1), 1)
        test_string(self, aae)
