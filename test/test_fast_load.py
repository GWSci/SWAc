import unittest
from swacmod.input_output import fast_load
import swacmod.utils as u

class Test_Fast_Load_Returns_Dictonaries(unittest.TestCase):
    def test_returned_object_is_dictinary(self):
        data = load_data()
        self.assertEqual(dict, type(data))
    
    def test_returned_object_contains_dictinaries(self):
        data = load_data()
        for key in data.keys():
            self.assertEqual(dict, type(data[key]))
    
def load_data():
        data = fast_load(u.CONSTANTS["SPECS_FILE"], u.CONSTANTS["INPUT_FILE"])
        return data    