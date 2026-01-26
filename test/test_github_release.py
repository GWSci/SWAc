import unittest
from github_release import convert_version_to_tag_name

class Test_Github_Release(unittest.TestCase):
    def test_convert_version_to_tag_name(self):
        self.assertEqual("v0.0.0", convert_version_to_tag_name([0, 0, 0]))
        self.assertEqual("v0.0.1", convert_version_to_tag_name([0, 0, 1]))