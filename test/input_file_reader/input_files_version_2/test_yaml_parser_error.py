import unittest
from swacmod.input_files.input_files_version_2.loader import load_yaml
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource

class Test_YAML_Errors_In_Input_Files(unittest.TestCase):
    def test_yaml_file_with_misaligned_keys(self):
        filename = 'recipe.yml'
        file_contents = 'ingredients:\n   sausage: 1\n  potato: 2\n' # missing one space before potato
        mock_file_opener = MockFileResource.make_mock_file_opener({filename: file_contents})
        with self.assertRaisesRegex(Exception, 'Please see guidelines for the YAML syntax'):
            load_yaml(filename, mock_file_opener)
    
    def test_yaml_file_missing_space_between_key_and_value(self):
        filename = 'recipe.yml'
        file_contents = 'ingredients:\n   sausage:1\n   potato: 2\n' # missing one space between potato and 1
        mock_file_opener = MockFileResource.make_mock_file_opener({filename: file_contents})
        with self.assertRaisesRegex(Exception, 'Please see guidelines for the YAML syntax'):
            load_yaml(filename, mock_file_opener)
    
    def test_yaml_file_missing_colon_between_key_and_value(self):
        filename = 'recipe.yml'
        file_contents = 'ingredients:\n   sausage 1\n   potato: 2\n' # missing colon between potato and 1
        mock_file_opener = MockFileResource.make_mock_file_opener({filename: file_contents})
        with self.assertRaisesRegex(Exception, 'Please see guidelines for the YAML syntax'):
            load_yaml(filename, mock_file_opener)
