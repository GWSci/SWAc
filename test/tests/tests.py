import unittest
from swacmod import utils as u
from swacmod.input_files.input_files_version_1 import validation as v
from swacmod.input_files.input_files_version_1 import input_data as input_data

class EndToEndTests(unittest.TestCase):
    def test_val_num_nodes_with_wrong_type(self):
        name = 'num_nodes'
        data = {'params': {name: 1.0}, 'specs': {name: {'required': True, 'type': [int]}}}
        self.assertRaises(u.ValidationError, v.val_num_nodes, data, name)

    def test_val_num_nodes_when_out_of_range(self):
        name = 'num_nodes'
        data = {'params': {name: -1}, 'specs': {name: {'required': True, 'type': [int]}}}
        self.assertRaises(u.ValidationError, v.val_num_nodes, data, name)

    def test_val_start_date(self):
        name = 'start_date'
        data = {'params': {name: 1.0}, 'specs': {name :{'required': True, 'type': [int]}}}
        self.assertRaises(u.ValidationError, v.val_start_date, data, name)
