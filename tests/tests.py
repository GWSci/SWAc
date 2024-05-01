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
from swacmod import validation as v
from swacmod import input_output as io
import swacmod.timer as timer

###############################################################################
class EndToEndTests(unittest.TestCase):
    """Test suite for the SWAcMod project."""

    specs_file = u.CONSTANTS['SPECS_FILE']
    input_file = u.CONSTANTS['TEST_INPUT_FILE']
    input_dir = u.CONSTANTS['TEST_INPUT_DIR']

    data = io.load_and_validate(specs_file, input_file, input_dir)
    if not data:
        print('Loading failed, interrupting tests now.')
        sys.exit()

    ids = range(1, data['params']['num_nodes'] + 1)

    def test_keys(self):
        """Test for validate_all() function."""

        actual_keys = set(self.data['series'].keys()) | set(self.data['params'].keys())

        generated_keys = set([
            'date', 'months', 'kc_list', 'macro_prop', 'macro_limit',
            'macro_act', 'macro_rec', 'ror_prop', 'ror_limit'])
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

    @unittest.skip
    def test_get_output(self):
        """Test for get_output() function."""
        time_switcher = timer.make_time_switcher()
        for node in self.ids:
            output = swacmod.get_output(self.data, node, time_switcher)
            results = io.load_results()
            for key in u.CONSTANTS['COL_ORDER']:
                if key in ['', 'date']:
                    continue
                self.assertTrue(key in results)
                self.assertTrue(key in self.data['series'] or
                                key in output)
                self.assertEqual(len(results) + 2,
                                 len(output))
                if key in self.data['series'] and key not in \
                        ['rainfall_ts', 'pe_ts', 'swabs_ts',
                         'swdis_ts']:
                    types = (list, np.ndarray)

                    if isinstance(self.data['series'][key][0], types):
                        new_list = [i[0] for i in self.data['series'][key]]
                    else:
                        new_list = self.data['series'][key]
                else:
                    new_list = output[key]

                self.assertEqual(len(new_list), len(results[key]))
            for num in range(len(self.data['series']['date'])):
                for key in u.CONSTANTS['COL_ORDER']:
                    if key in ['', 'date']:
                        continue
                    try:
                        self.assertAlmostEqual(results[key][num],
                                               output[key][num],
                                               places=5)
                    except AssertionError as err:
                        print('\n')
                        for col in u.CONSTANTS['COL_ORDER']:
                            if col in ['', 'date']:
                                continue
                            if round(output[col][num] * 1e5) != \
                                    round(results[col][num] * 1e5):
                                print('%s: %s (%s)' % (col, output[col][num],
                                                       results[col][num]))
                        print('\n---> Failed at "%s", row %d\n' % (key, num))
                        raise AssertionError(err)


###############################################################################
if __name__ == '__main__':

    unittest.main()
