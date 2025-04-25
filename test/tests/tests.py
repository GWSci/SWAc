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
        """Test for val_num_nodes() function."""
        data = load_data()
        name = 'num_nodes'
        old = data['params'][name]
        data['params'][name] = 1.0
        self.assertRaises(u.ValidationError, v.val_num_nodes, data, name)
        data['params'][name] = -1
        self.assertRaises(u.ValidationError, v.val_num_nodes, data, name)
        data['params'][name] = old

    def test_val_start_date(self):
        """Test for val_start_date() function."""
        data = load_data()
        name = 'start_date'
        old = data['params'][name]
        data['params'][name] = 1.0
        self.assertRaises(u.ValidationError, v.val_start_date, data, name)
        data['params'][name] = old
