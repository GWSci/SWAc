import unittest
import swacmod.input_files.input_files_version_2.validator as validator

class Test_Specs(unittest.TestCase):
    def test_validate_matches_spec_when_field_not_in_spec_is_error(self):
        params = None
        specs = {}
        actual = validator.validate_matches_spec(specs, params, "aardvark")
        self.assert_one_error(actual, params, "some error")

    def assert_one_error(self, actual, params, expected_error):
        self.assertEqual(params, actual.data)
