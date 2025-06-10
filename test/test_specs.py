import unittest
import swacmod.input_files.input_files_version_2.validator as validator
from swacmod.input_files.input_files_version_2.specs import Input_Parameter

class Test_Specs(unittest.TestCase):
    def test_validate_spec_returns_error_when_key_is_not_in_spec(self):
        params = {}
        specs_object = []
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'aardvark'. Please contact support.")

    def test_validate_spec_returns_error_when_key_is_not_in_spec_2(self):
        params = {}
        specs_object = []
        actual = matches_spec_adaptor(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'bat'. Please contact support.")

    def test_validate_spec_returns_error_when_key_is_not_in_spec_3(self):
        params = {}
        specs_object = [Input_Parameter("aardvark", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'bat'. Please contact support.")

    def test_validate_spec_returns_error_when_required_key_is_missing(self):
        params = {}
        specs_object = [Input_Parameter("aardvark", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: The required key 'aardvark' could not be found.")

    def test_validate_spec_returns_error_when_required_key_is_missing_2(self):
        params = {}
        specs_object = [Input_Parameter("bat", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: The required key 'bat' could not be found.")

    def test_validate_spec_returns_no_errors_when_a_non_required_field_is_missing(self):
        params = {}
        specs_object = [Input_Parameter("aardvark", False, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    def test_validate_spec_returns_no_errors_when_a_non_required_field_is_present(self):
        params = {"aardvark": "bat"}
        specs_object = [Input_Parameter("aardvark", False, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    def test_validate_spec_returns_no_errors_when_a_required_field_is_present(self):
        params = {"aardvark": "bat"}
        specs_object = [Input_Parameter("aardvark", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    def test_validate_spec_returns_error_when_string_type_does_not_match(self):
        params = {"aardvark": 3}
        specs_object = [Input_Parameter("aardvark", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: The field 'aardvark' must be a string. The value '3' is invalid.")

    def test_validate_spec_returns_error_when_string_type_does_not_match(self):
        params = {"bat": 5.7}
        specs_object = [Input_Parameter("bat", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "bat")
        self.assert_one_error(
            actual,
            params,
            "Error: The field 'bat' must be a string. The value '5.7' is invalid.")

    def test_validate_spec_returns_no_error_when_string_type_matches(self):
        params = {"aardvark": "bat"}
        specs_object = [Input_Parameter("aardvark", True, [], [str], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    def test_validate_spec_returns_error_when_int_type_does_not_match_string(self):
        params = {"aardvark": "bat"}
        specs_object = [Input_Parameter("aardvark", True, [], [int], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: The field 'aardvark' must be an integer. The value 'bat' is invalid.")

    def test_validate_spec_returns_error_when_int_type_does_not_match_float(self):
        params = {"aardvark": 3.5}
        specs_object = [Input_Parameter("aardvark", True, [], [int], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: The field 'aardvark' must be an integer. The value '3.5' is invalid.")

    def test_validate_spec_returns_no_error_when_int_type_matches(self):
        params = {"aardvark": 7}
        specs_object = [Input_Parameter("aardvark", True, [], [int], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    def test_validate_spec_returns_error_when_int_type_matches(self):
        params = {"aardvark": 3}
        specs_object = [Input_Parameter("aardvark", True, [], [int], None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    def assert_one_error(self, actual, params, expected_error):
        self.assertEqual(params, actual.data)
        self.assertEqual([], actual.warnings)
        self.assertEqual([expected_error], actual.errors)

    def assert_no_errors(self, actual, params):
        self.assertEqual(params, actual.data)
        self.assertEqual([], actual.warnings)
        self.assertEqual([], actual.errors)

def matches_spec_adaptor(specs_object, params, key):
    return validator.validate_matches_spec(specs_object, params, key)