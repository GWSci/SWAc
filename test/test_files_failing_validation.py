import unittest
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource
from swacmod.input_files.input_files_version_2.input_data import load_and_validate

class Test_Files_Failing_Validation(unittest.TestCase):
    def test_load_and_validate_a_model_with_a_jumbled_up_file(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['time_periods'][-1][-1] = '043.potato'
        input_file_contents = ""
        for k, v in input_file.items():
            input_file_contents += f"{k}: {v}\n"
        mock_file_open = MockFileResource.make_mock_file_opener({'input.yml': input_file_contents})
        mock_input_directory = MockFileResource.make_sample_input_directory()
        mock_filename_exists = MockFileResource.make_mock_filename_exists(mock_input_directory)        
        parsed_input_data = load_and_validate(input_file='input.yml', input_dir='', file_opener=mock_file_open, filename_exists=mock_filename_exists)
        self.assertIn('time_periods', '\n'.join(parsed_input_data.errors))