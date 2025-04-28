import unittest
import swacmod.input_files.input_files_version_2.input_data as input_data

class Test_Input_Data_v2_Validation(unittest.TestCase):
    def test_wording_for_required_key_missing(self):
        params = {}
        input_file = "some_input_file.yml"
        input_data.validate(params, input_file)
        self.assertEqual(1, 1)
