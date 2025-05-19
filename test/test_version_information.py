import unittest
import swacmod.version_information as version_information
from _version import version
from _commit_id import commit_id
import io
from build import update_version

class Test_Version_Information(unittest.TestCase):
    def test_version_information_has_the_correct_version_number(self):
        expected = f'{version[0]}.{version[1]}.{version[2]}'
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
    
    def test_version_information_has_the_correct_commit_id(self):
        expected = str(commit_id)
        actual = version_information.format_version_information()
        self.assertIn(expected, actual)
    
    def test_build_script_updates_the_version(self):
        filename = 'mock_version.py'
        contents = 'version = [1,0,0]'
        file_open = make_mock_file_opener({filename: contents})
        update_version(filename=filename, file_open=file_open)






def make_mock_file_opener(contents):
        return lambda filename: io.StringIO(contents[filename])


