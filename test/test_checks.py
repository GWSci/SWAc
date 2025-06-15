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

    def test_validating_keys_when_one_key_substituted(self):
        self.assert_validate_keys_fails({"c":1}, [dict], ["a"])

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
        self.assert_validate_keys_fails([{"c":1}], [list, dict], ["a"])
        self.assert_validate_keys_fails([{"c":1}, {"a":1}], [list, dict], ["a"])
        self.assert_validate_keys_fails([{"a":1}, {"c":1}], [list, dict], ["a"])

    def assert_validate_keys_passes(self, param, t_types, keys):
        _validate_keys_adaptor(param, t_types, keys)

    def assert_validate_keys_fails(self, param, t_types, keys):
        with self.assertRaises(u.ValidationError):
            _validate_keys_adaptor(param, t_types, keys)

    def test_list_length_when_len_list_is_none(self):
        self.assert_validate_list_length_passes([], [list], None)
        self.assert_validate_list_length_passes(["a"], [list], None)

    def test_list_length_when_len_list_is_empty(self):
        self.assert_validate_list_length_passes([], [list], [])
        self.assert_validate_list_length_passes(["a"], [list], [])

    def test_list_length_when_len_list_length_matches(self):
        self.assert_validate_list_length_passes([], [list], [0])
        self.assert_validate_list_length_passes(["a"], [list], [1])
        self.assert_validate_list_length_passes(["a", "b"], [list], [2])

    def test_list_length_when_len_list_length_differs_and_expected_length_is_zero(self):
        # TODO Possible bug when expected list length is zero.
        #  This test shows that if the value of expected length is zero,
        #  then it will allow a list of non-zero length. This is surprising and
        #  it is unclear whether this is to deliberately allow for any list
        #  length, or whether it is a bug.
        self.assert_validate_list_length_passes(["a"], [list], [0])

    def test_list_length_when_len_list_length_differs(self):
        self.assert_validate_list_length_fails([], [list], [1])
        self.assert_validate_list_length_fails(["a", "b"], [list], [1])
        self.assert_validate_list_length_fails([], [list], [2])
        self.assert_validate_list_length_fails(["a"], [list], [2])

    def assert_validate_list_length_passes(self, param, t_types, list_lengths):
        _validate_list_length_adaptor(param, t_types, list_lengths)

    def assert_validate_list_length_fails(self, param, t_types, list_lengths):
        with self.assertRaises(u.ValidationError):
            _validate_list_length_adaptor(param, t_types, list_lengths)

def _validate_keys_adaptor(param, t_types, keys):
    checks.check_type(param=param, name="cat", t_types=t_types, len_list=None, keys=keys)

def _validate_list_length_adaptor(param, t_types, list_lengths):
    checks.check_type(param=param, name="cat", t_types=t_types, len_list=list_lengths, keys=None)
