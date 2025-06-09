import unittest
import swacmod.input_files.input_files_version_2.validator as validator

class Test_Specs(unittest.TestCase):
    def test_x(self):
        params = None
        specs = None
        actual = validator.validate_matches_spec(specs, params, "aardvark")
        self.assertEqual(1, 1)
