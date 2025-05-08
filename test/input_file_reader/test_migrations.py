import unittest
import swacmod.input_files.input_file_reader as input_file_reader

class Test_Migrations(unittest.TestCase):
    def test_migrating_v1_to_v2_updates_the_version(self):
        data = {'params':{"version": 1}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertEqual(2, actual["params"]["version"])

    def test_migrating_all_versions_migrates_to_latest(self):
        expected = 2
        self.assertEqual(expected, input_file_reader.migrate({'params': {"version": 2}})["params"]["version"])
        self.assertEqual(expected, input_file_reader.migrate({'params': {"version": 1}})["params"]["version"])
        self.assertEqual(expected, input_file_reader.migrate({'params': {}})["params"]["version"])

    def test_migrating_from_leakage_process_to_subroot_leakage_process(self):
        data = {'params':{'leakage_process': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('subroot_leakage_process', actual['params'])
    
    def test_migrating_from_subsoilzone_leakage_fraction_to_subroot_leakage_fraction(self):
        data = {'params':{'subsoilzone_leakage_fraction': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('subroot_leakage_fraction', actual['params'])
    
    def test_migrating_from_sw_process_natproc_to_sw_ponding_process(self):
        data = {'params':{'sw_process_natproc': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('sw_ponding_process', actual['params'])
    
    def test_migtating_from_historical_nitrate_process_to_historical_solute_process(self):
        data = {'params':{'historical_nitrate_process': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('historical_solute_process', actual['params'])

    def test_migtating_from_nitrate_process_to_solute_process(self):
        data = {'params':{'nitrate_process': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_process', actual['params'])
    
    def test_migtating_from_nitrate_calibration_a_to_solute_calibration_a(self):
        data = {'params':{'nitrate_calibration_a': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_calibration_a', actual['params'])
    
    def test_migtating_from_nitrate_calibration_mu_to_solute_calibration_mu(self):
        data = {'params':{'nitrate_calibration_mu': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_calibration_mu', actual['params'])
    
    def test_migtating_from_nitrate_calibration_sigma_to_solute_calibration_sigma(self):
        data = {'params':{'nitrate_calibration_sigma': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_calibration_sigma', actual['params'])
    
    def test_migtating_from_nitrate_calibration_alpha_to_solute_calibration_alpha(self):
        data = {'params':{'nitrate_calibration_alpha': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_calibration_alpha', actual['params'])
    
    def test_migtating_from_nitrate_calibration_effective_porosity_to_solute_calibration_effective_porosity(self):
        data = {'params':{'nitrate_calibration_effective_porosity': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_calibration_effective_porosity', actual['params'])

    def test_migtating_from_nitrate_depth_to_water_to_solute_depth_to_water(self):
        data = {'params':{'nitrate_depth_to_water': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_depth_to_water', actual['params'])
    
    def test_migtating_from_nitrate_loading_to_solute_loading(self):
        data = {'params':{'nitrate_loading': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('solute_loading', actual['params'])
