import unittest
import swacmod.input_files.input_files_version_2.validator as validator

class Test_Specs(unittest.TestCase):
    def test_validate_matches_spec_returns_error_when_key_is_not_in_spec(self):
        params = {}
        specs = {}
        actual = validator.validate_matches_spec(specs, params, "aardvark")
        self.assert_one_error(
            actual,
            params,
            "Error: No spec was found for the key 'aardvark'. Please contact support.")

    def assert_one_error(self, actual, params, expected_error):
        self.assertEqual(params, actual.data)
        self.assertEqual([], actual.warnings)
        self.assertEqual([expected_error], actual.errors)
