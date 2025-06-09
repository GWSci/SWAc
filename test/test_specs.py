import unittest
import swacmod.input_files.input_files_version_2.validator as validator
from swacmod.input_files.input_files_version_2.specs import Input_Parameter

class Test_Specs(unittest.TestCase):
    def test_validate_matches_spec_returns_error_when_key_is_not_in_spec(self):
        params = {}
        specs_object = []
        actual = validator.validate_matches_spec(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'aardvark'. Please contact support.")

    def test_validate_matches_spec_returns_error_when_key_is_not_in_spec_2(self):
        params = {}
        specs_object = []
        actual = validator.validate_matches_spec(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'bat'. Please contact support.")

    def test_validate_matches_spec_returns_error_when_key_is_not_in_spec_3(self):
        params = {}
        specs_object = [Input_Parameter("aardvark", required=True, alt_format=[], type=[str], constraints=None)]
        actual = validator.validate_matches_spec(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'bat'. Please contact support.")

    def test_validate_matches_spec_returns_error_when_required_key_is_missing(self):
        params = {}
        specs_object = [Input_Parameter("aardvark", required=True, alt_format=[], type=[str], constraints=None)]
        actual = validator.validate_matches_spec(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: The required key 'aardvark' could not be found.")

    def test_validate_matches_spec_returns_error_when_required_key_is_missing_2(self):
        params = {}
        specs_object = [Input_Parameter("bat", required=True, alt_format=[], type=[str], constraints=None)]
        actual = validator.validate_matches_spec(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: The required key 'bat' could not be found.")

    def assert_one_error(self, actual, params, expected_error):
        self.assertEqual(params, actual.data)
        self.assertEqual([], actual.warnings)
        self.assertEqual([expected_error], actual.errors)
