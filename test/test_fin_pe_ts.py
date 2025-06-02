import unittest
import swacmod.utils as u
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource
from swacmod.input_files.input_files_version_2.input_data import load_and_validate as load_and_validate

class Test_pe_ts_Finalisation(unittest.TestCase):
    def test_canopy_and_fao_are_disabled(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'disabled'
        input_file['canopy_process'] = 'disabled'
        parsed_input_data = make_parsed_data(input_file)
        self.assertFalse(parsed_input_data.has_errors())

    def test_canopy_is_disabled_and_fao_is_enabled(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'enabled'
        input_file['canopy_process'] = 'disabled'
        parsed_input_data = make_parsed_data(input_file)
        self.assertFalse(parsed_input_data.has_errors())

    def test_canopy_is_enabled_and_fao_is_disabled(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'disabled'
        input_file['canopy_process'] = 'enabled'
        parsed_input_data = make_parsed_data(input_file)
        self.assertFalse(parsed_input_data.has_errors())

def make_parsed_data(input_file):
    input_file_contents = ""
    for k, v in input_file.items():
        input_file_contents += f"{k}: {v}\n"
    mock_file_open = MockFileResource.make_mock_file_opener({'input.yml': input_file_contents})
    mock_input_directory = MockFileResource.make_sample_input_directory()
    mock_filename_exists = MockFileResource.make_mock_filename_exists(mock_input_directory)
    parsed_input_data = load_and_validate(input_file='input.yml', input_dir='', file_opener=mock_file_open, filename_exists=mock_filename_exists)
    return parsed_input_data