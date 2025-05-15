import unittest
import swacmod.version_information as version_information
from swacmod import __version__

class Test_Version_Information(unittest.TestCase):
    def test_version_information_has_the_correct_version_number(self):
        expected = __version__
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
