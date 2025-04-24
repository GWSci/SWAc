import unittest
from swacmod.input_data import fast_load
import swacmod.utils as u
import swacmod.checks as c
import swacmod.finalization as f
import random

class Test_Fast_Load_Returns_Dictonaries(unittest.TestCase):
    def test_returned_object_is_dictinary(self):
        data = load_test_data()
        self.assertEqual(dict, type(data))
    
    def test_returned_object_contains_dictinaries(self):
        data = load_test_data()
        for key in data.keys():
            self.assertEqual(dict, type(data[key]))

class Test_Fast_Load_And_Check(unittest.TestCase):
    def test_random_required_parameter_is_missing(self):
        data = load_test_data()
        f.finalize_required_params(data)
        data = remove_random_required_param(data)
        try:
            # this should fail with a check error
            c.check_required(data)
            actual = 1
        except:
            actual = 0
        expected = 0
        self.assertEqual(expected, actual)

def get_random_required_key(d):
    key = random.choice(list(d.keys()))
    while not d[key]['required']:
        key = random.choice(list(d.keys()))
    return key

def remove_random_required_param(data):
    key = get_random_required_key(data['specs'])
    while key not in list(data['params'].keys()) or key == 'temp_file_backed_array_directory': #this is not a user-defined required param
        key = get_random_required_key(data['specs'])
    data['params'].pop(key)
    return data

def load_test_data():
        data = fast_load(u.CONSTANTS["SPECS_FILE"], u.CONSTANTS["TEST_INPUT_FILE"])
        return data    