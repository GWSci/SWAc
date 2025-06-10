import unittest
import swacmod.input_files.input_files_version_2.validator as validator
from swacmod.input_files.input_files_version_2.specs import Input_Parameter
from datetime import date, datetime

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
        self._test_error_message_when_type_of_value_is_not_ok(
            [str],
            3,
            "Error: The field 'aardvark' must be a string. The value '3' is invalid.")

    def test_validate_spec_returns_error_when_string_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [str],
            3.5,
            "Error: The field 'aardvark' must be a string. The value '3.5' is invalid.")

    def test_validate_spec_returns_error_when_int_type_does_not_match_string(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [int],
            "bat",
            "Error: The field 'aardvark' must be an integer. The value 'bat' is invalid.")

    def test_validate_spec_returns_error_when_int_type_does_not_match_float(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [int],
            3.5,
            "Error: The field 'aardvark' must be an integer. The value '3.5' is invalid.")

    def test_validate_spec_returns_error_when_bool_type_does_not_match_float(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [bool],
            3.5,
            "Error: The field 'aardvark' must be a boolean. The value '3.5' is invalid.")

    def test_validate_spec_returns_error_when_float_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [float],
            "x",
            "Error: The field 'aardvark' must be a float. The value 'x' is invalid.")

    def test_validate_spec_returns_error_when_date_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [date],
            "x",
            "Error: The field 'aardvark' must be a date. The value 'x' is invalid.")

    def _test_error_message_when_type_of_value_is_not_ok(
            self, spec_type, value, expected_error_message):
        params = {"aardvark": value}
        specs_object = [Input_Parameter("aardvark", True, [], spec_type, None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            expected_error_message)

    def test_validate_spec_returns_no_error_when_string_type_matches(self):
        self._test_no_error_when_type_of_value_is_ok([str], "bat")

    def test_validate_spec_returns_no_error_when_int_type_matches(self):
        self._test_no_error_when_type_of_value_is_ok([int], 7)

    def test_validate_spec_returns_no_error_when_bool_type_matches(self):
        self._test_no_error_when_type_of_value_is_ok([bool], True)
        self._test_no_error_when_type_of_value_is_ok([bool], False)

    def test_validate_spec_returns_no_error_when_float_type_matches(self):
        self._test_no_error_when_type_of_value_is_ok([float], 3.5)

    def test_validate_spec_returns_no_error_when_float_type_matches_int(self):
        self._test_no_error_when_type_of_value_is_ok([float], 7)

    def test_validate_spec_returns_no_error_when_date_type_matches(self):
        self._test_no_error_when_type_of_value_is_ok([date], date(2025, 12, 30))

    def test_validate_spec_returns_no_error_when_datetime_type_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [datetime], datetime(2025, 12, 30))

    def _test_no_error_when_type_of_value_is_ok(self, spec_type, value):
        params = {"aardvark": value}
        specs_object = [Input_Parameter("aardvark", True, [], spec_type, None)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        self.assert_no_errors(actual, params)

    # TODO Test type=[datetime]
    # TODO Test type=[dict, float]
    # TODO Test type=[dict, int]
    # TODO Test type=[dict, list, float]
    # TODO Test type=[dict, list, int]
    # TODO Test type=[dict, list]
    # TODO Test type=[dict, str]
    # TODO Test type=[list, dict, list]
    # TODO Test type=[list, list, float]
    # TODO Test type=[list, list, int]
    # TODO Test type=[set, int]
    # TODO Test type=None

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