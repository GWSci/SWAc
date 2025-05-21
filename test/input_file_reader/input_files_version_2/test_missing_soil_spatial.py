import unittest
import swacmod.input_files.input_files_version_2.input_data as input_data_v2
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource

class Test_Input_File_With_Missing_Soil_Spatial(unittest.TestCase):
    def test_is_missing_and_is_finalised_successfully(self):
        input_filename = 'potato.yml'
        input_dir = ''
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['soil_spatial']
        input_file_contents = ""
        for k, v in input_file.items():
            input_file_contents += f"{k}: {v}\n"
        mock_file_open = MockFileResource.make_mock_file_opener({input_filename: input_file_contents})
        parsed_input_data = input_data_v2.load_and_validate(input_filename, input_dir, mock_file_open)
        data = parsed_input_data.data
        self.assertIsNot(data['params']['soil_spatial'], int)