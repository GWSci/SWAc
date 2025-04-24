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
