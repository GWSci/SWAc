import unittest
import io
import swacmod.input_files.input_files_version_2.input_data as input_data

class Test_Input_Files_v2(unittest.TestCase):
    def test_load_yaml_reads_empty_yaml_file(self):
        self.assert_load_yaml("", None)

    def assert_load_yaml(self, input_file_contents, expected):
        file_opener = make_mock_file_opener({
            "aardvark.yaml": input_file_contents,
        })
        actual = input_data.load_yaml("aardvark.yaml", file_opener=file_opener)
        self.assertEqual(expected, actual)

    def test_load_yaml_reads_yaml_file_with_one_key(self):
        self.assert_load_yaml("bat: cat", {"bat": "cat"})

    def test_load_yaml_converts_keys_to_lower_case(self):
        self.assert_load_yaml("BAT: CAT", {"bat": "CAT"})

    def test_load_yml_alt_format_when_there_are_no_extra_keys(self):
        params = load_alt_yaml_adaptor("infiltration_limit", """
infiltration_limit:
  1: 1.2
  2: 3.4
""")
        expected = {"infiltration_limit": {1: 1.2, 2: 3.4}}
        self.assertEqual(expected, params)

    def test_load_yml_alt_format_when_the_requested_key_is_missing(self):
        with self.assertRaisesRegex(Exception, 'Error: Could not find the key "infiltration_limit" in the file "aardvark.yaml".'):
            load_alt_yaml_adaptor("infiltration_limit", "")

def load_alt_yaml_adaptor(param, external_file_contents):
    file_opener = make_mock_file_opener({
        "aardvark.yaml": external_file_contents,
    })
    params = {param: "aardvark.yaml"}
    absolute = "aardvark.yaml"
    input_data.load_yml_alt_format(params, param, absolute, file_opener)
    return params

def make_mock_file_opener(filenames_to_contents):
    return lambda filename: io.StringIO(filenames_to_contents[filename])
