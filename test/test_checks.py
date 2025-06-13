import unittest

import swacmod.input_files.input_files_version_2.checks as checks

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assert_validate_keys_passes("aardvark", [str], None)

    def assert_validate_keys_passes(self, param, t_types, keys):
        checks.check_type(param=param, name="cat", t_types=t_types, len_list=None, keys=keys)
