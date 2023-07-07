import unittest

from utils.my_typing import String


def test_string(self: unittest.TestCase, obj: String):
    self.assertIsNotNone(obj.string)
    print(obj.string)
