import unittest
import swacmod.version_information as version_information
from _version import version
from _commit_id import commit_id

class Test_Version_Information(unittest.TestCase):
    def test_version_information_has_the_correct_version_number(self):
        expected = version
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
    
    def test_version_information_has_the_correct_commit_id(self):
        expected = commit_id
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
