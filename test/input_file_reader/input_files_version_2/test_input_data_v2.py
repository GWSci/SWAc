import unittest
import io
import swacmod.input_files.input_files_version_2.input_data as input_data

class Test_Input_Files_v2(unittest.TestCase):
    def test_load_yaml_reads_empty_yaml_file(self):
        file_opener = make_mock_file_opener({
            "aardvark.yaml": "",
        })
        actual = input_data.load_yaml("aardvark.yaml", file_opener=file_opener)
        self.assertIsNone(actual)

    def test_load_yaml_reads_yaml_file_with_one_key(self):
        file_opener = make_mock_file_opener({
            "aardvark.yaml": "bat: cat",
        })
        expected = {"bat": "cat"}
        actual = input_data.load_yaml("aardvark.yaml", file_opener=file_opener)
        self.assertEqual(expected, actual)

    def test_load_yaml_converts_keys_to_lower_case(self):
        file_opener = make_mock_file_opener({
            "aardvark.yaml": "BAT: CAT",
        })
        expected = {"bat": "CAT"}
        actual = input_data.load_yaml("aardvark.yaml", file_opener=file_opener)
        self.assertEqual(expected, actual)

def make_mock_file_opener(filenames_to_contents):
    return lambda filename: io.StringIO(filenames_to_contents[filename])
