import unittest
import swacmod.version_information as version_information
from _version import version
from _commit_id import commit_id
import io
import build

class Test_Version_Information(unittest.TestCase):
    def test_version_information_has_the_correct_version_number(self):
        expected = f'{version[0]}.{version[1]}.{version[2]}'
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
    
    def test_version_information_has_the_correct_commit_id(self):
        expected = str(commit_id)
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
    
    def test_build_script_gets_the_old_version(self):
        filename = 'mock_version.py'
        contents = 'version = [1, 0, 0]'
        file_open = make_mock_file_opener({filename: contents})
        old_version = build.get_old_version(filename, file_open)
        self.assertIn(str(old_version), contents)
    
    def test_build_script_updates_the_version(self):
        filename = 'mock_version.py'
        contents = 'version = [1, 0, 0]'
        file_open = make_mock_file_opener({filename: contents})
        old_version = build.get_old_version(filename, file_open)
        new_version = build.update_version(old_version)
        self.assertEqual([1, 0, 1], new_version)






def make_mock_file_opener(filename_contents):
        return lambda filename, how : io.StringIO(filename_contents[filename])


