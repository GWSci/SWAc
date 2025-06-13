import unittest
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.checks as checks

class MyTestCase(unittest.TestCase):
    def test_validating_keys_when_keys_is_none(self):
        self.assert_validate_keys_passes({}, [dict], None)
        self.assert_validate_keys_passes({"a":1}, [dict], None)

    def assert_validate_keys_passes(self, param, t_types, keys):
        checks.check_type(param=param, name="cat", t_types=t_types, len_list=None, keys=keys)

    def assert_validate_keys_fails(self, param, t_types, keys):
        with self.assertRaises(u.ValidationError):
            checks.check_type(param=param, name="cat", t_types=t_types, len_list=None, keys=keys)
