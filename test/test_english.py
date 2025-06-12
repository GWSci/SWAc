import unittest
import swacmod.input_files.input_files_version_2.english as english

class Test_English(unittest.TestCase):
    def test_format_list(self):
        self.assertEqual("", english.format_list([]))
        self.assertEqual("'1'", english.format_list([1]))
