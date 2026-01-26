import unittest
import github_release

class Test_Github_Release(unittest.TestCase):
    def test_x(self):
        input_version = [0, 0, 0]
        expected = "v0.0.0"
        actual = github_release.convert_version_to_tag_name(input_version)
        self.assertEqual(expected, actual)