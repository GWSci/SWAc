import unittest
from build import convert_sys_platform_to_platform

class Test_Build(unittest.TestCase):
    def test_x(self):
        self.assertEqual("aardvark", convert_sys_platform_to_platform("aardvark"))
        self.assertEqual("linux", convert_sys_platform_to_platform("linux"))
