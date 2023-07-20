import unittest

from utils.tests import test_string
from .downsample import Downsample
from .upsample import Upsample

from .encoder import Encoder2Normal
from .decoder import Decoder
from .discriminator import Discriminator
from .cat_discriminator import CatDiscriminator


class TestString(unittest.TestCase):
    """ Tests if all subclasses of String define the `string` attribute """
    def test_downsample(self):
        downsample = Downsample(1)
        test_string(self, downsample)

    def test_upsample(self):
        upsample = Upsample(1, [1])
        test_string(self, upsample)

    def test_encoder(self):
        encoder = Encoder2Normal([10, 10], 1)
        test_string(self, encoder)

    def test_decoder(self):
        decoder = Decoder(1, [10, 10])
        test_string(self, decoder)

    def test_discriminator(self):
        discriminator = Discriminator(1)
        test_string(self, discriminator)

    def test_cat_discriminator(self):
        discriminator = CatDiscriminator((1, 1))
        test_string(self, discriminator)
