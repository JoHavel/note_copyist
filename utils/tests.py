import unittest

from utils.my_typing import String


def test_string(self: unittest.TestCase, obj: String):
    """ Test if class deriving `String` has defined `string` attribute (it cannot be done with abc, so we test it) """
    self.assertIsNotNone(obj.string)
    print(obj.string)
