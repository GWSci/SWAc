import unittest
import swacmod.input_files.input_files_version_2.input_data as input_data

class Test_Input_Data_v2_Validation(unittest.TestCase):
    def test_a_valid_file_has_no_errors(self):
        params = {}
        input_file = "some_input_file.yml"
        validation_result = input_data.validate(params, input_file)
        self.assertEqual(0, len(validation_result.errors))
