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

class Test_YAML_Errors_In_Main_Input_File(unittest.TestCase):
    def test_main_yaml_input_file_with_misaligned_keys(self):
        filename = 'cheesecake.yml'
        params = MockFileResource.make_sample_valid_input_file()
        file_contents = ""
        for k, v in params.items():
            if k == "num_nodes":
                file_contents += f" {k}: {v}\n" # one space before the key
            else:
                file_contents += f"{k}: {v}\n"
        mock_file_opener = MockFileResource.make_mock_file_opener({filename: file_contents})
        with self.assertRaisesRegex(Exception, 'Please see guidelines for the YAML syntax'):
            load_yaml(filename, mock_file_opener)

    def test_main_yaml_input_file_with_missing_space_between_key_and_value(self):
        filename = 'cheesecake.yml'
        params = MockFileResource.make_sample_valid_input_file()
        file_contents = ""
        for k, v in params.items():
            if k == "num_nodes":
                file_contents += f"{k}:{v}\n" # one space before the key
            else:
                file_contents += f"{k}: {v}\n"
        mock_file_opener = MockFileResource.make_mock_file_opener({filename: file_contents})
        with self.assertRaisesRegex(Exception, 'Please see guidelines for the YAML syntax'):
            load_yaml(filename, mock_file_opener)
