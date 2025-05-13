import unittest
import io
import swacmod.input_files.input_files_version_2.input_data as input_data
import swacmod.input_files.input_files_version_2.validator as validator
import swacmod.input_files.input_files_version_2.specs as specs_module
import swacmod.input_files.input_file_reader as input_file_reader
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource

class Printer_Spy:
    def __init__(self):
        self.result = ""
    
    def do_print(self, x):
        self.result += x

class Test_Input_Data_v2_Validation_Through_Input_File_Reader(unittest.TestCase):
    def test_reading_input_data_with_an_invalid_file_throws_an_exception(self):
        input_file = "some_file.yml"
        input_dir = "some_dir/"
        params = MockFileResource.make_sample_valid_input_file()
        del params["num_nodes"]
        input_file_contents = ""
        for k, v in params.items():
            input_file_contents += f"{k}: {v}\n"
        file_opener = MockFileResource.make_mock_file_opener({input_file: input_file_contents})
        printer = lambda x: None
        with self.assertRaisesRegex(Exception, "Run has exited with errors."):
            input_file_reader.read_inputs(None, input_file, input_dir, file_opener = file_opener, printer=printer)

    def test_reading_input_data_with_an_invalid_file_records_the_error(self):
        input_file = "some_file.yml"
        input_dir = "some_dir/"
        params = MockFileResource.make_sample_valid_input_file()
        del params["num_nodes"]
        input_file_contents = ""
        for k, v in params.items():
            input_file_contents += f"{k}: {v}\n"
        file_opener = MockFileResource.make_mock_file_opener({input_file: input_file_contents})
        printer_spy = Printer_Spy()
        try:
            input_file_reader.read_inputs(None, input_file, input_dir, file_opener=file_opener, printer=printer_spy.do_print)
        except:
            pass
        self.assertEqual('Error: The file "some_file.yml" is missing the required field "num_nodes".', printer_spy.result)

    def test_reading_a_file_when_a_filename_doesnt_exist_throws_an_exception(self):
        input_file = "some_file.yml"
        input_dir = ""
        params = MockFileResource.make_sample_valid_input_file()
        input_file_contents = ""
        for k, v in params.items():
            input_file_contents += f"{k}: {v}\n"
        file_opener = MockFileResource.make_mock_file_opener({input_file: input_file_contents})
        printer = lambda x: None
        mock_dir = MockFileResource.make_sample_input_directory()
        mock_dir.remove("time_periods.csv")
        filename_exists = MockFileResource.make_mock_filename_exists(mock_dir)
        with self.assertRaisesRegex(Exception, "Run has exited with errors."):
            input_file_reader.read_inputs(None, input_file, input_dir, file_opener = file_opener, printer=printer, filename_exists=filename_exists)

    def test_reading_a_file_when_a_filename_doesnt_exist_records_the_error(self):
        input_file = "some_file.yml"
        input_dir = ""
        params = MockFileResource.make_sample_valid_input_file()
        params['time_periods'] = 'potato.csv'
        input_file_contents = ""
        for k, v in params.items():
            input_file_contents += f"{k}: {v}\n"
        file_opener = MockFileResource.make_mock_file_opener({input_file: input_file_contents})
        printer_spy = Printer_Spy()
        mock_dir = MockFileResource.make_sample_input_directory()
        filename_exists = MockFileResource.make_mock_filename_exists(mock_dir)
        try:
            input_file_reader.read_inputs(None, input_file, input_dir, file_opener=file_opener, printer=printer_spy.do_print, filename_exists=filename_exists)
        except:
            pass
        self.assertEqual('Error: Unknown file name: "potato.csv"', printer_spy.result)

class Test_Input_Data_v2_Validation(unittest.TestCase):
    def test_a_valid_file_has_no_errors(self):
        specs = specs_module.make_specs()
        params = MockFileResource.make_sample_valid_input_file()
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        self.assertEqual(0, len(validation_result.errors))

    def test_a_valid_file_has_no_warnings(self):
        specs = specs_module.make_specs()
        params = MockFileResource.make_sample_valid_input_file()
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        self.assertEqual(0, len(validation_result.warnings))

    def test_validating_a_file_missing_a_required_field_reports_an_error(self):
        specs = specs_module.make_specs()
        params = MockFileResource.make_sample_valid_input_file()
        del params["version"]
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        self.assertEqual(1, len(validation_result.errors))

    def test_validating_a_file_missing_a_required_field_has_text_describing_the_problem(self):
        specs = specs_module.make_specs()
        params = MockFileResource.make_sample_valid_input_file()
        del params["version"]
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        expected = 'Error: The file "some_input_file.yml" is missing the required field "version".'
        actual = validation_result.errors[0]
        self.assertEqual(expected, actual)

    def test_validating_empty_params_reports_all_missing_required_parameters(self):
        specs = specs_module.make_specs()
        params = {}
        validation_result = validator.validate_keys(specs, params, "some_input_file.yml")
        all_errors_string = "\n".join(validation_result.errors)
        self.assertIn('"version"', all_errors_string)
        self.assertIn('"run_name"', all_errors_string)

    def test_no_errors_when_all_filenames_exist(self):
        specs = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = MockFileResource.make_sample_valid_input_file()
        validation_result = input_data._validate_filenames(params, specs, "", filename_always_exists)
        self.assertEqual(0, len(validation_result.errors))

    def test_no_warnings_when_all_filenames_exist(self):
        specs = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = MockFileResource.make_sample_valid_input_file()
        validation_result = input_data._validate_filenames(params, specs, "", filename_always_exists)
        self.assertEqual(0, len(validation_result.warnings))
    
    def test_x(self):
        specs = specs_module.make_specs_dictionary(specs_module.make_specs())
        params = MockFileResource.make_sample_valid_input_file()
        validation_result = input_data._validate_filenames(params, specs, "", filename_never_exists)
        all_errors_string = "\n".join(validation_result.errors)
        self.assertIn('Error: Unknown file name:', all_errors_string)

def filename_always_exists(filename):
    return True

def filename_never_exists(filename):
    return False
