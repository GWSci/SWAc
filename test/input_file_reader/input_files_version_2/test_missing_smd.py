import unittest
from swacmod_run import run
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.input_data as input_data_v2
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource
from test.dummy_environment import Dummy_Environment

class Test_Input_File_With_Missing_SMD(unittest.TestCase):
    def test_smd_is_missing_and_is_finalised_successfully_as_0(self):
        input_filename = 'potato.yml'
        input_dir = ''
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['smd']
        input_file_contents = ""
        for k, v in input_file.items():
            input_file_contents += f"{k}: {v}\n"
        mock_file_open = MockFileResource.make_mock_file_opener({input_filename: input_file_contents})
        parsed_input_data = input_data_v2.load_and_validate(input_filename, input_dir, mock_file_open)
        data = parsed_input_data.data
        expected = [0. for zone,smd in enumerate(data['params']['smd']['starting_SMD'])]
        actual = data['params']['smd']['starting_SMD']
        self.assertEqual(expected,actual)

    def test_smd_is_missing_and_model_runs(self):
        filename = 'potato.yml'
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['smd']
        input_file_contents = ""
        for k, v in input_file.items():
            input_file_contents += f"{k}: {v}\n"
        mock_file_open = MockFileResource.make_mock_file_opener({filename: input_file_contents})
        default_input_file = u.CONSTANTS["INPUT_FILE"]
        try:
            u.CONSTANTS["INPUT_FILE"] = filename
            run(test=False, skip=True, env=Dummy_Environment(), file_opener=mock_file_open)
        finally:
             u.CONSTANTS["INPUT_FILE"] = default_input_file

