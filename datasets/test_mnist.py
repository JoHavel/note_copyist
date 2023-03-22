import unittest
from datasets import mnist


class TestMnist(unittest.TestCase):
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
            b'\xb2r\x9e\xb0\t2\xbc\x19\xce\xba\xf4\xc1x\x89\xf1\x91',
            hashlib.md5(mnist.X_train.tobytes()).digest()
        )
        self.assertEqual(
            b'\xd4\x1d\x8c\xd9\x8f\x00\xb2\x04\xe9\x80\t\x98\xec\xf8B~',
            hashlib.md5(mnist.X_val.tobytes()).digest()
        )
        self.assertEqual(
            b'\x1fd\xb2\xb2\xcf|;\xf4q\xd6\xfc\xb3u\xe01\xfb',
            hashlib.md5(mnist.X_test.tobytes()).digest()
        )
