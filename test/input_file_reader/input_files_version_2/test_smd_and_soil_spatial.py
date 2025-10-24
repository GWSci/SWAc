import unittest
from swacmod_run import run
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.input_data as input_data_v2
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource
from test.dummy_environment import Dummy_Environment
import swacmod.input_files.input_files_version_2.finalization as f
import swacmod.input_files.input_files_version_2.validation_new as validation
import swacmod.input_files.input_files_version_2.loader as loader
import swacmod.input_files.input_files_version_2.specs as specs_module

class Test_Input_File_With_Missing_SMD(unittest.TestCase):
    def test_smd_is_missing_and_is_finalised_successfully_as_0(self):
        filename = 'potato.yml'
        input_dir = ''
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['smd']
        mock_file_open = make_mock_file_open_and_contents(filename, input_file)
        parsed_input_data = input_data_v2.load_and_validate(filename, input_dir, mock_file_open)
        data = parsed_input_data.data
        expected = [0. for zone,smd in enumerate(data['params']['smd']['starting_SMD'])]
        actual = data['params']['smd']['starting_SMD']
        self.assertEqual(expected,actual)

    def test_soil_zone_names_and_soil_spatial_are_missing_and_smd_is_still_finalised_as_0(self):
        filename = 'potato.yml'
        input_dir = ''
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['smd']
        del input_file['soil_spatial']
        del input_file['soil_zone_names']
        mock_file_open = make_mock_file_open_and_contents(filename, input_file)
        parsed_input_data = input_data_v2.load_and_validate(filename, input_dir, mock_file_open)
        data = parsed_input_data.data
        expected = [0. for zone,smd in enumerate(data['params']['smd']['starting_SMD'])]
        actual = data['params']['smd']['starting_SMD']
        self.assertEqual(expected,actual)

    def test_smd_is_missing_and_model_runs(self):
        filename = 'potato.yml'
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['smd']
        mock_file_open = make_mock_file_open_and_contents(filename, input_file)
        default_input_file = u.CONSTANTS["INPUT_FILE"]
        try:
            u.CONSTANTS["INPUT_FILE"] = filename
            run(test=False, skip=True, env=Dummy_Environment(), file_opener=mock_file_open)
        finally:
             u.CONSTANTS["INPUT_FILE"] = default_input_file

class Test_Input_File_With_Missing_Soil_Spatial(unittest.TestCase):
    def test_is_missing_and_is_finalised_correctly(self):
        params = {'fao_process': 'disabled',
                  'soil_zone_names': {1: 'zone1', 2: 'zone2'}, 
                  'soil_spatial': None,
                  'num_nodes': 2}
        data = {'params': params}
        f.fin_soil_spatial(data, 'soil_spatial')
        expected = {1: [0.5, 0.5], 2: [0.5, 0.5]}
        actual = data['params']['soil_spatial']
        self.assertEqual(expected, actual)

    def test_soil_spatial_is_missing_and_is_finalised_successfully(self):
        filename = 'potato.yml'
        input_dir = ''
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['soil_spatial']
        mock_file_open = make_mock_file_open_and_contents(filename, input_file)
        parsed_input_data = input_data_v2.load_and_validate(filename, input_dir, mock_file_open)
        data = parsed_input_data.data
        self.assertIsNot(data['params']['soil_spatial'], None)
    
    def test_soil_spatial_is_missing_and_model_runs(self):
        filename = 'potato.yml'
        input_file = MockFileResource.make_sample_input_file_with_data()
        del input_file['smd']
        mock_file_open = make_mock_file_open_and_contents(filename, input_file)
        default_input_file = u.CONSTANTS["INPUT_FILE"]
        try:
            u.CONSTANTS["INPUT_FILE"] = filename
            run(test=False, skip=True, env=Dummy_Environment(), file_opener=mock_file_open)
        finally:
             u.CONSTANTS["INPUT_FILE"] = default_input_file

class Test_Always_Validate_SMD(unittest.TestCase):
    def test_smd_is_validated_when_fao_process_is_enabled(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'enabled'
        data = mess_up_parameter_and_make_data(input_file, 'smd')
        errors = []
        validation.val_smd(errors, data, 'smd')
        self.assertEqual(1, len(errors))

    def test_smd_is_validated_when_fao_process_is_disabled(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'disabled'
        data = mess_up_parameter_and_make_data(input_file, 'smd')
        errors = []
        validation.val_smd(errors, data, 'smd')
        self.assertEqual(1, len(errors))

class Test_Always_Validate_Soil_Spatial(unittest.TestCase):
    def test_soil_spatal_is_validated_when_fao_process_is_enabled_and_fao_input_is_ls(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'enabled'
        input_file['fao_input'] = 'ls'
        data = mess_up_parameter_and_make_data(input_file, 'soil_spatial')
        errors = []
        validation.val_soil_spatial(errors, data, 'soil_spatial')
        self.assertEqual(1, len(errors))
    
    def test_soil_spatal_is_validated_when_fao_process_is_enabled_and_fao_input_is_l(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'enabled'
        input_file['fao_input'] = 'l'
        data = mess_up_parameter_and_make_data(input_file, 'soil_spatial')
        errors = []
        validation.val_soil_spatial(errors, data, 'soil_spatial')
        self.assertEqual(1, len(errors))
    
    def test_soil_spatal_is_validated_when_fao_process_is_disabled_and_fao_input_is_ls(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'disabled'
        input_file['fao_input'] = 'ls'
        data = mess_up_parameter_and_make_data(input_file, 'soil_spatial')
        errors = []
        validation.val_soil_spatial(errors, data, 'soil_spatial')
        self.assertEqual(1, len(errors))

    def test_soil_spatal_is_validated_when_fao_process_is_enabled_and_fao_input_is_l_2(self):
        input_file = MockFileResource.make_sample_input_file_with_data()
        input_file['fao_process'] = 'disabled'
        input_file['fao_input'] = 'l'
        data = mess_up_parameter_and_make_data(input_file, 'soil_spatial')
        errors = []
        validation.val_soil_spatial(errors, data, 'soil_spatial')
        self.assertEqual(1, len(errors))

def mess_up_parameter_and_make_data(input_file, param):
    input_file[param] = {}
    mock_file_open = make_mock_file_open_and_contents('potato.yml', input_file)
    params = loader.load_yaml('potato.yml', mock_file_open)
    specs = specs_module.make_specs_dictionary(specs_module.make_specs())
    data = {'params':params, 'specs':specs}
    return data

def make_mock_file_open_and_contents(filename, input_file):
    input_file_contents = ""
    for k, v in input_file.items():
        input_file_contents += f"{k}: {v}\n"
    return MockFileResource.make_mock_file_opener({filename: input_file_contents})