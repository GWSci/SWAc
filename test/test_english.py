import unittest
import swacmod.input_files.input_files_version_2.english as english

class Test_English(unittest.TestCase):
    def test_x(self):
        self.assertEqual("", english.format_list([]))
