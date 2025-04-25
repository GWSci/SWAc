#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""SWAcMod tests."""

# Standard Library
import unittest


# Internal modules
from swacmod import utils as u
from swacmod.input_files.input_files_version_1 import validation as v
from swacmod.input_files.input_files_version_1 import input_data as input_data

class EndToEndTests(unittest.TestCase):
    """Test suite for the SWAcMod project."""

    specs_file = u.CONSTANTS['SPECS_FILE']
    input_file = u.CONSTANTS['TEST_INPUT_FILE']
    input_dir = u.CONSTANTS['TEST_INPUT_DIR']

    data = input_data.load_and_validate(specs_file, input_file, input_dir)

    def test_val_num_nodes(self):
        """Test for val_num_nodes() function."""
        name = 'num_nodes'
        old = self.data['params'][name]
        self.data['params'][name] = 1.0
        self.assertRaises(u.ValidationError, v.val_num_nodes, self.data, name)
        self.data['params'][name] = -1
        self.assertRaises(u.ValidationError, v.val_num_nodes, self.data, name)
        self.data['params'][name] = old

    def test_val_start_date(self):
        """Test for val_start_date() function."""
        name = 'start_date'
        old = self.data['params'][name]
        self.data['params'][name] = 1.0
        self.assertRaises(u.ValidationError, v.val_start_date, self.data, name)
        self.data['params'][name] = old

