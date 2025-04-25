import unittest
import io
import swacmod.input_files.input_files_version_2.input_data as input_data

class Test_Input_Files_v2(unittest.TestCase):
    def test_load_yaml_reads_empty_yaml_file(self):
        mock_file_system = {
            "aardvark.yaml": "",
        }
        file_opener = lambda filename: io.StringIO(mock_file_system[filename])
        actual = input_data.load_yaml("aardvark.yaml", file_opener=file_opener)
        self.assertIsNone(actual)
