import unittest
import swacmod.input_files.input_files_version_2.validation_new as validation_new
import swacmod.input_files.input_files_version_2.specs as specs_module

class Test_Validation_New(unittest.TestCase):
    def test_validation_converts_exceptions_to_error_list(self):
        spec_maps = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = {"run_name": 5}
        actual = validation_new.validate(params, spec_maps)
        self.assertEqual(
            "---> Validation failed: Parameter \"run_name\" has to be a string, found a <class 'int'> instead",
            actual[0])
