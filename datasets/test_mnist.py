import unittest
from datasets import mnist


class TestMnist(unittest.TestCase):
    """ Test if MNIST is properly prepared """
    def test_lengths(self):
        self.assertEqual(mnist.train_len, len(mnist.X_train))
        self.assertEqual(mnist.val_len, len(mnist.X_val))
        self.assertEqual(mnist.test_len, len(mnist.X_test))

        self.assertEqual(mnist.train_len, len(mnist.y_train))
        self.assertEqual(mnist.val_len, len(mnist.y_val))
        self.assertEqual(mnist.test_len, len(mnist.y_test))

    def test_uniformity(self):
        # FIXME MNIST isn't uniform?
        for i in range(10):
            self.assertEqual(mnist.train_len/10, len([y for y in mnist.y_train if y == i]))
        for i in range(10):
            self.assertEqual(mnist.val_len/10, len([y for y in mnist.y_val if y == i]))
        for i in range(10):
            self.assertEqual(mnist.test_len/10, len([y for y in mnist.y_test if y == i]))

    def test_reproducibility(self):
        import hashlib
        self.assertEqual(
            b'\xca\x8d\x87\xf7\xa8L\x99\x19k\xf8^\xc1\xdf\x9d\x83T',
            hashlib.md5(mnist.X_train.tobytes()).digest()
        )
        self.assertEqual(
            b'_\xa3iGs2N\x91$>\x82\xa7\x08\x94"\x97',
            hashlib.md5(mnist.X_val.tobytes()).digest()
        )
        self.assertEqual(
            b'\x1fd\xb2\xb2\xcf|;\xf4q\xd6\xfc\xb3u\xe01\xfb',
            hashlib.md5(mnist.X_test.tobytes()).digest()
        )
