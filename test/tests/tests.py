#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""SWAcMod tests."""

# Standard Library
import sys
import unittest

# Third Party Libraries
import numpy as np

# Internal modules
import swacmod_run as swacmod
from swacmod import utils as u
from swacmod.input_files.input_files_version_1 import validation as v
from swacmod import input_output as io
from swacmod.input_files.input_files_version_1 import input_data as input_data
import swacmod.timer as timer

class EndToEndTests(unittest.TestCase):
    """Test suite for the SWAcMod project."""

    specs_file = u.CONSTANTS['SPECS_FILE']
    input_file = u.CONSTANTS['TEST_INPUT_FILE']
    input_dir = u.CONSTANTS['TEST_INPUT_DIR']

    data = input_data.load_and_validate(specs_file, input_file, input_dir)
    if not data:
        print('Loading failed, interrupting tests now.')
        sys.exit()

    ids = range(1, data['params']['num_nodes'] + 1)

    def test_keys(self):
        """Test for validate_all() function."""

        actual_keys = set(self.data['series'].keys()) | set(self.data['params'].keys())

        generated_keys = set([
            'date',
            'months',
            'kc_list',
            'macro_prop',
            'macro_limit',
            'macro_act',
            'macro_rec',
            'ror_prop',
            'ror_limit',
            'sw_direct_rech',
            'sw_bed_infiltn',
            'sw_pe_to_open_wat',
            'sw_activ',
            'ror_act',
            'sw_pond_area',
            'sw_downstr'])
        expected_keys = set(self.data['specs']) | generated_keys

        self.assertEqual(expected_keys, actual_keys)

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

