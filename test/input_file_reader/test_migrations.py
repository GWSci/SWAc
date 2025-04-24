import unittest
import swacmod.input_files.input_file_reader as input_file_reader

class Test_Migrations(unittest.TestCase):
    def test_migrating_v1_to_v2_updates_the_version(self):
        data = {"version": 1}
        actual = input_file_reader.migrate_v1_to_v2(data)
