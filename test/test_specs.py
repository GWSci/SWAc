import unittest
import swacmod.input_files.input_files_version_2.validator as validator
from swacmod.input_files.input_files_version_2.specs import Input_Parameter
from datetime import date, datetime
import numpy as np

class Test_Specs(unittest.TestCase):
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

    def test_validate_spec_returns_no_errors_when_a_non_required_field_is_None(self):
        params = {"aardvark": None}
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

    def test_validate_spec_returns_error_when_datetime_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [datetime],
            "x",
            "Error: The field 'aardvark' must be a datetime. The value 'x' is invalid.")

    def test_validate_spec_returns_error_when_set_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [set, int],
            "x",
            "Error: The field 'aardvark' must be a set of integers. The value 'x' is invalid.")

    def test_validate_spec_returns_error_when_set_type_does_not_match_2(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [set, str],
            "x",
            "Error: The field 'aardvark' must be a set of strings. The value 'x' is invalid.")

    def test_validate_spec_returns_error_when_set_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [set, int],
            {"a", "b", "c"},
            "Error: The field 'aardvark' must be a set of integers. The member 'a' is invalid.")

    def test_validate_spec_returns_error_when_set_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [set, int],
            {1, "b", 3},
            "Error: The field 'aardvark' must be a set of integers. The member 'b' is invalid.")

    def test_validate_spec_returns_error_when_dict_type_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, int],
            "x",
            "Error: The field 'aardvark' must be a dictionary mapping integers to integers. The value 'x' is invalid.")

    @unittest.skip(
            reason="Validating the keys is more complicated than this."
                + "Sometimes they are ints. Other times they are strings and "
                + "this is not currently encoded inthe specs object.")
    def test_validate_spec_returns_error_when_dict_type_key_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, int],
            {1: 11, "y": 13, 3: 17},
            "Error: The field 'aardvark' must be a dictionary mapping integers to integers. The dictionary key 'y' is invalid.")

    def test_validate_spec_returns_error_when_dict_int_type_value_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, int],
            {1: 11, 2: "z", 3: 17},
            "Error: The field 'aardvark' must be a dictionary mapping integers to integers. The dictionary value 'z' is invalid.")

    def test_validate_spec_returns_error_when_dict_str_type_value_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, str],
            {1: "a", 2: 7, 3: "c"},
            "Error: The field 'aardvark' must be a dictionary mapping integers to strings. The dictionary value '7' is invalid.")

    def test_validate_spec_returns_error_when_dict_float_type_value_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, float],
            {1: 11.5, 2: 17, 3: "cat"},
            "Error: The field 'aardvark' must be a dictionary mapping integers to floats. The dictionary value 'cat' is invalid.")

    def test_validate_spec_returns_error_when_dict_list_int_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, list, int],
            "x",
            "Error: The field 'aardvark' must be a dictionary mapping integers to list of integers. The value 'x' is invalid.")

    def test_validate_spec_returns_error_when_dict_list_int_does_not_match_list_1(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, list, int],
            {1: [11], 2: "bat", 3: [13]},
            "Error: The field 'aardvark' must be a dictionary mapping integers to list of integers. The dictionary value 'bat' is invalid.")

    def test_validate_spec_returns_error_when_dict_list_int_does_not_match_list_2(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, list, int],
            {1: [11], 2: ["cat"], 3: [13]},
            "Error: The field 'aardvark' must be a dictionary mapping integers to list of integers. The inner list value 'cat' is invalid.")

    def test_validate_spec_returns_error_when_dict_list_int_does_not_match_list_3(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, list, int],
            {1: [11], 2: np.array([11.2]), 3: [13]},
            "Error: The field 'aardvark' must be a dictionary mapping integers to list of integers. The inner list value '11.2' is invalid.")

    def test_validate_spec_returns_error_when_dict_list_int_does_not_match_inner_value(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, list, float],
            {1: [11], 2: ["cat"], 3: [13]},
            "Error: The field 'aardvark' must be a dictionary mapping integers to list of floats. The inner list value 'cat' is invalid.")

    def test_validate_spec_returns_error_when_dict_list_does_not_match_value(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [dict, list],
            {1: [11], 2: ["cat"], 3: 13},
            "Error: The field 'aardvark' must be a dictionary mapping integers to lists. The dictionary value '13' is invalid.")

    def test_validate_spec_returns_error_when_list_list_int_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, list, int],
            "cat",
            "Error: The field 'aardvark' must be a list of list of integers. The value 'cat' is invalid.")

    def test_validate_spec_returns_error_when_list_list_int_does_not_match_member(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, list, int],
            [[1, 2], [3, 5], "bat"],
            "Error: The field 'aardvark' must be a list of list of integers. The member 'bat' is invalid.")

    def test_validate_spec_returns_error_when_list_list_int_does_not_match_member(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, list, int],
            [[1, 2], [3, 5], [7, "dog"]],
            "Error: The field 'aardvark' must be a list of list of integers. The inner list value 'dog' is invalid.")

    def test_validate_spec_returns_error_when_list_list_float_does_not_match_member(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, list, float],
            [[1, 2], [3.4, 5.6], [7.0, "dog"]],
            "Error: The field 'aardvark' must be a list of list of floats. The inner list value 'dog' is invalid.")

    def test_validate_spec_returns_error_when_list_dict_list_does_not_match(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, dict, list],
            "elephant",
            "Error: The field 'aardvark' must be a list of dictionary of lists. The value 'elephant' is invalid.")

    def test_validate_spec_returns_error_when_list_dict_list_does_not_match_member(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, dict, list],
            [{"a": [1, "a"]}, {"a": [2, "b"]}, "goat"],
            "Error: The field 'aardvark' must be a list of dictionary of lists. The member 'goat' is invalid.")

    def test_validate_spec_returns_error_when_list_dict_list_does_not_match_inner(self):
        self._test_error_message_when_type_of_value_is_not_ok(
            [list, dict, list],
            [{"a": [1, "a"]}, {"a": [2, "b"]}, {"a": "hippopotamus"}],
            "Error: The field 'aardvark' must be a list of dictionary of lists. The inner list value 'hippopotamus' is invalid.")

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

    def test_validate_spec_returns_no_error_when_set_of_ints_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [set, int], {3, 5, 7})

    def test_validate_spec_returns_no_error_when_set_of_strings_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [set, str], {"a", "b", "c"})

    def test_validate_spec_returns_no_error_when_set_of_floats_matches_floats_and_integers(self):
        self._test_no_error_when_type_of_value_is_ok(
            [set, float], {1.1, 2, 3.3})

    def test_validate_spec_returns_no_error_when_dict_of_ints_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, int], {1: 11, 2: 13, 3: 17})

    def test_validate_spec_returns_no_error_when_dict_of_ints_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, float], {1: 11.1, 2: 13.3, 3: 17})

    def test_validate_spec_returns_no_error_when_dict_list_int_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list, int], {1: [11], 2: [13], 3: [17]})

    def test_validate_spec_returns_no_error_when_dict_list_int_matches_2(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list, int],
            {1: np.array([11]), 2: np.array([13]), 3: np.array([17])})

    def test_validate_spec_returns_no_error_when_dict_list_float_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list, float], {1: [11.1], 2: [13.1], 3: [17.1]})

    def test_validate_spec_returns_no_error_when_dict_list_float_matches_2(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list, float],
            {1: np.array([11.0]), 2: np.array([13.0]), 3: np.array([17.0])})

    def test_validate_spec_returns_no_error_when_dict_list_float_matches_3(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list, float], {1: [11], 2: [13], 3: [17]})

    def test_validate_spec_returns_no_error_when_dict_list_float_matches_4(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list, float],
            {1: np.array([11]), 2: np.array([13]), 3: np.array([17])})

    def test_validate_spec_returns_no_error_when_dict_list_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [dict, list], {1: [11.3], 2: [13], 3: ["cat"]})

    def test_validate_spec_returns_no_error_when_list_list_int_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, list, int], [[1, 2], [3, 5], [7, 11]])

    def test_validate_spec_returns_no_error_when_list_list_int_matches_2(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, list, int], np.array([[1, 2], [3, 5], [7, 11]]))

    def test_validate_spec_returns_no_error_when_list_list_float_matches(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, list, float], [[1, 2], [3, 5], [7, 11]])

    def test_validate_spec_returns_no_error_when_list_list_float_matches_2(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, list, float], [[1.1, 2.1], [3.1, 5.1], [7.1, 11.1]])

    def test_validate_spec_returns_no_error_when_list_list_float_matches_3(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, list, float], np.array([[1.1, 2.1], [3.1, 5.1], [7.1, 11.1]]))

    def test_validate_spec_returns_no_error_when_list_list_float_matches_4(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, list, float], np.array([[1, 2], [3, 5], [7, 11]]))

    def test_validate_spec_returns_no_error_when_list_dict_list_matches_(self):
        self._test_no_error_when_type_of_value_is_ok(
            [list, dict, list],
            [{"a": [1, "a"]}, {"a": [2, "b"]}, {"c": [3, "c"]}])

    def test_validate_spec_no_error_when_constraints_are_none(self):
        params, actual = self.validate_constraints(None, "bat")
        self.assert_no_errors(actual, params)

    def test_validate_spec_no_error_when_value_matches_constraints(self):
        params, actual = self.validate_constraints(["bat", "cat"], "bat")
        self.assert_no_errors(actual, params)

    def test_validate_spec_no_error_when_value_matches_constraints_2(self):
        params, actual = self.validate_constraints(["bat", "cat"], "cat")
        self.assert_no_errors(actual, params)

    def test_validate_spec_error_when_value_does_not_match_constraints(self):
        params, actual = self.validate_constraints(["bat", "cat"], "dog")
        self.assert_one_error(
            actual,
            params,
            "Error: The field 'aardvark' has the invalid value 'dog'. It must be one of: 'bat' or 'cat'.")

    def test_validate_spec_error_when_value_does_not_match_constraints2(self):
        params, actual = self.validate_constraints(["fox", "goat"], "dog")
        self.assert_one_error(
            actual,
            params,
            "Error: The field 'aardvark' has the invalid value 'dog'. It must be one of: 'fox' or 'goat'.")

    def test_validate_all_fields(self):
        params = {
            "aardvark": 1,
            "cat": "x",
        }
        specs_object = [
            Input_Parameter("aardvark", True, [], [str], None),
            Input_Parameter("bat", True, [], [str], None),
            Input_Parameter("cat", True, [], [str], None),
        ]
        actual = validator.validate_all_match_spec(specs_object, params)
        expected = [
            "Error: The field 'aardvark' must be a string. The value '1' is invalid.",
            "Error: The required key 'bat' could not be found."
        ]
        self.assertEqual(expected, actual.errors)

    def test_validate_all_fields_allowing_alt(self):
        params = {
            "aardvark": 1,
            "bat": "dog.csv",
            "cat": "elephant"
        }
        specs_object = [
            Input_Parameter("aardvark", True, ['yml', 'csv'], [int], None),
            Input_Parameter("bat", True, ['yml', 'csv'], [int], None),
            Input_Parameter("cat", True, ['yml', 'csv'], [int], None),
        ]
        actual = validator.validate_all_match_spec_allowing_alt(specs_object, params)
        expected = [
            "Error: The field 'cat' must be an integer. The value 'elephant' is invalid.",
        ]
        self.assertEqual(expected, actual.errors)

    def test_validate_spatial_output_date_is_None(self):
        params, actual = self.validate_spatial_output_date(None)
        actual = self.assert_no_errors(actual, params)

    def test_validate_spatial_output_date_is_mean(self):
        params, actual = self.validate_spatial_output_date("mean")
        actual = self.assert_no_errors(actual, params)

    def test_validate_spatial_output_date_is_date(self):
        params, actual = self.validate_spatial_output_date(date(2025, 12, 30))
        actual = self.assert_no_errors(actual, params)

    def test_validate_spatial_output_date_is_mean(self):
        params, actual = self.validate_spatial_output_date(123)
        actual = self.assert_one_error(
            actual,
            params,
            "Error: The field 'spatial_output_date' must be a date or the string 'none'. The value '123' is invalid.")

    def validate_spatial_output_date(self, value):
        params = {"spatial_output_date": value}
        specs_object = [Input_Parameter(
                'spatial_output_date',
                required=False,
                alt_format=[],
                type=["spatial_output_date"],
                constraints=None)]
        actual = matches_spec_adaptor(
            specs_object, params, "spatial_output_date")
            
        return params,actual

    def validate_constraints(self, constraints, value):
        params = {"aardvark": value}
        specs_object = [Input_Parameter("aardvark", True, [], [str], constraints)]
        actual = matches_spec_adaptor(specs_object, params, "aardvark")
        return params, actual

    def _test_no_error_when_type_of_value_is_ok(self, spec_type, value):
        params = {"aardvark": value}
        specs_object = [Input_Parameter("aardvark", True, [], spec_type, None)]
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
    spec = specs_object[0]
    return validator.validate_matches_spec(spec, params, key)