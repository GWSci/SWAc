import unittest
import swacmod.input_files.input_file_reader as input_file_reader

class Test_Migrations(unittest.TestCase):
    def test_migrating_v1_to_v2_updates_the_version(self):
        data = {"version": 1}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertEqual(2, actual["version"])

    def test_migrating_all_versions_migrates_to_latest(self):
        expected = 2
        self.assertEqual(expected, input_file_reader.migrate({"version": 2})["version"])
        self.assertEqual(expected, input_file_reader.migrate({"version": 1})["version"])
        self.assertEqual(expected, input_file_reader.migrate({})["version"])

    def test_migrating_from_leakage_process_to_subroot_leakage_process(self):
        data = {'params':{'leakage_process': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('subroot_leakage_process', actual['params'])
    
    def test_migrating_from_subsoilzone_leakage_fraction_to_subroot_leakage_fraction(self):
        data = {'params':{'subsoilzone_leakage_fraction': 'sausage'}}
        actual = input_file_reader.migrate_v1_to_v2(data)
        self.assertIn('subroot_leakage_fraction', actual['params'])

