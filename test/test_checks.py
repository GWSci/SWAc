import unittest
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.checks as checks

class MyTestCase(unittest.TestCase):
    def test_validating_keys_when_keys_is_none(self):
        self.assert_validate_keys_passes({}, [dict], None)
        self.assert_validate_keys_passes({"a":1}, [dict], None)

    def test_validating_keys_when_keys_is_empty(self):
        self.assert_validate_keys_passes({}, [dict], None)
        self.assert_validate_keys_passes({"a":1}, [dict], [])

    def test_validating_keys_when_keys_matches_exactly(self):
        self.assert_validate_keys_passes({"a":1}, [dict], ["a"])
        self.assert_validate_keys_passes({"a":1, "b": 2}, [dict], ["a", "b"])

    def test_validating_keys_when_one_key_missing(self):
        self.assert_validate_keys_fails({}, [dict], ["a"])
        self.assert_validate_keys_fails({"a":1,}, [dict], ["a", "b"])
        self.assert_validate_keys_fails({"b":1,}, [dict], ["a", "b"])

    def test_validating_keys_when_extra_key(self):
        self.assert_validate_keys_passes({"a":1, "b":2}, [dict], ["a"])
        self.assert_validate_keys_passes({"a":1, "b":2, "c":3}, [dict], ["a", "b"])

    def test_validating_keys_when_type_is_not_dict(self):
        self.assert_validate_keys_passes("aardvark", [str], ["a"])

    def test_validating_keys_when_dict_is_nested_in_list_matches_exactly(self):
        self.assert_validate_keys_passes([], [list, dict], ["a"])
        self.assert_validate_keys_passes([{"a":1}], [list, dict], ["a"])
        self.assert_validate_keys_passes([{"a":1}, {"a":1}], [list, dict], ["a"])

    def test_validating_keys_when_dict_is_nested_in_list_deos_not_match(self):
        self.assert_validate_keys_fails([{}], [list, dict], ["a"])

    def assert_validate_keys_passes(self, param, t_types, keys):
        checks.check_type(param=param, name="cat", t_types=t_types, len_list=None, keys=keys)

    def assert_validate_keys_fails(self, param, t_types, keys):
        with self.assertRaises(u.ValidationError):
            checks.check_type(param=param, name="cat", t_types=t_types, len_list=None, keys=keys)
