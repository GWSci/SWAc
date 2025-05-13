import unittest
import swacmod.input_files.input_files_version_2.validator as validator
import swacmod.input_files.input_files_version_2.input_data as input_data
import swacmod.input_files.input_files_version_2.loader as loader
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource

class Test_Input_Files_v2(unittest.TestCase):
    def test_load_yaml_reads_empty_yaml_file(self):
        self.assert_load_yaml("", None)

    def assert_load_yaml(self, input_file_contents, expected):
        file_opener = MockFileResource.make_mock_file_opener({
            "aardvark.yaml": input_file_contents,
        })
        actual = loader.load_yaml("aardvark.yaml", file_opener=file_opener)
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

    def test_load_yml_alt_format_when_extra_keys_are_present(self):
        with self.assertRaisesRegex(Exception, 'Error: Found the key "cat" in the file "aardvark.yaml". The only key should be "infiltration_limit". Common causes of this error are typos and forgetting to indent the key-value pairs in a map.'):
            load_alt_yaml_adaptor("infiltration_limit", """
infiltration_limit: bat
cat: dog
""")

    def test_load_yml_alt_format_when_multiple_errors_are_present(self):
        with self.assertRaisesRegex(Exception, 'infiltration_limit.*\n.*cat.*\n.*elephant'):
            load_alt_yaml_adaptor("infiltration_limit", """
cat: dog
elephant: fox
""")

    def test_validate_no_extra_params_when_params_are_empty_finds_no_errors(self):
        specs = {"aardvark": None, "bat": None}
        params = {}
        actual = validator.validate_no_extra_params(specs, params, "input.yaml")
        self.assertEqual(0, len(actual.errors))

    def test_validate_no_extra_params_when_params_are_recognised_finds_no_errors(self):
        specs = {"aardvark": None, "bat": None}
        params = {"aardvark": None, "bat": None}
        actual = validator.validate_no_extra_params(specs, params, "input.yaml")
        self.assertEqual(0, len(actual.errors))

    def test_validate_no_extra_params_when_one_params_is_unrecognised_finds_an_error(self):
        specs = {"aardvark": None, "bat": None}
        params = {"aardvark": None, "bat": None, "cat": None}
        actual = validator.validate_no_extra_params(specs, params, "input.yaml")
        self.assertEqual(1, len(actual.errors))
        self.assertIn(
            'Error: Found the key "cat" in the file "input.yaml". This key is not valid. A full list of valid keys is: ["aardvark", "bat"].',
            actual.errors)

    def test_validate_no_extra_params_when_multiple_params_are_unrecognised_finds_2_errors(self):
        specs = {"aardvark": None, "bat": None}
        params = {"aardvark": None, "bat": None, "dog": None, "cat": None}
        actual = validator.validate_no_extra_params(specs, params, "input.yaml")
        self.assertEqual(2, len(actual.errors))
        self.assertIn(
            'Error: Found the key "cat" in the file "input.yaml". This key is not valid. A full list of valid keys is: ["aardvark", "bat"].',
            actual.errors)
        self.assertIn(
            'Error: Found the key "dog" in the file "input.yaml". This key is not valid. A full list of valid keys is: ["aardvark", "bat"].',
            actual.errors)

def load_alt_yaml_adaptor(param, external_file_contents):
    file_opener = MockFileResource.make_mock_file_opener({
        "aardvark.yaml": external_file_contents,
    })
    params = {param: "aardvark.yaml"}
    absolute = "aardvark.yaml"
    input_data.load_yml_alt_format(params, param, absolute, file_opener)
    return params

