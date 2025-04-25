import unittest
from swacmod import utils as u
from swacmod.input_files.input_files_version_1 import validation as v
from swacmod.input_files.input_files_version_1 import input_data as input_data

def load_data():
    specs_file = u.CONSTANTS['SPECS_FILE']
    input_file = u.CONSTANTS['TEST_INPUT_FILE']
    input_dir = u.CONSTANTS['TEST_INPUT_DIR']

    data = input_data.load_and_validate(specs_file, input_file, input_dir)
    return data

class EndToEndTests(unittest.TestCase):
    def test_val_num_nodes(self):
        data = load_data()
        name = 'num_nodes'
        data['params'][name] = 1.0
        self.assertRaises(u.ValidationError, v.val_num_nodes, data, name)
        data['params'][name] = -1
        self.assertRaises(u.ValidationError, v.val_num_nodes, data, name)

    def test_val_start_date(self):
        data = load_data()
        name = 'start_date'
        data['params'][name] = 1.0
        self.assertRaises(u.ValidationError, v.val_start_date, data, name)
