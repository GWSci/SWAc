#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod tests."""

# Standard Library
import os
import sys
import unittest

# Third Party Libraries
import yaml
import numpy as np

# Internal modules
import swacmod_run as swacmod
from swacmod import utils as u
from swacmod import validation as v
from swacmod import input_output as io

##############################################################################
def generate_test_file(name, num_nodes):
    """Generate input files for tests."""
    if num_nodes < 1:
        print 'Parameter "num_nodes" has to be >= 1.'
        return

    path = os.path.join(u.CONSTANTS['TEST_INPUT_DIR'], name + '.yml')
    obj = io.load_yaml(path)
    comments = [i for i in open(path, 'r').readlines() if i.startswith('#')]

    fileout = open(path, 'w')
    for comment in comments:
        fileout.write('%s' % comment)

    fileout.write('%s:\n' % name)
    for node in range(1, num_nodes+1):
        fileout.write('    %d: %s\n' % (node, obj[name][1]))

    fileout.close()


###############################################################################
def generate_test_files(num_nodes=10):
    """Generate input files for tests."""
    if not isinstance(num_nodes, int) or not num_nodes >= 1:
        print 'Parameter "num_nodes" has to be >= 1.'
        return

    for name in ['node_areas', 'reporting_zone_mapping',
                 'rainfall_zone_mapping', 'pe_zone_mapping',
                 'temperature_zone_mapping', 'subroot_zone_mapping',
                 'rapid_runoff_zone_mapping', 'rorecharge_zone_mapping',
                 'macropore_zone_mapping', 'free_throughfall',
                 'max_canopy_storage', 'snow_params', 'interflow_params',
                 'subsoilzone_leakage_fraction', 'soil_spatial', 'lu_spatial',
                 'recharge_attenuation_params', 'sw_params']:
        generate_test_file(name, num_nodes)

    filein = open(u.CONSTANTS['TEST_INPUT_FILE'], 'r').readlines()
    fileout = open(u.CONSTANTS['TEST_INPUT_FILE'], 'w')
    for line in filein:
        if line.lower().startswith('num_nodes:'):
            new_line = 'num_nodes: %d\n' % num_nodes
        else:
            new_line = line
        fileout.write(new_line)
    fileout.close()


###############################################################################
def benchmark():
    """Run SWAcMod on a log scale of nodes."""
    print
    for num_nodes in [1, 10, 100, 1000, 10000]:
        print 'Running benchmark: %d nodes' % num_nodes
        generate_test_files(num_nodes=num_nodes)
        os.system('python -m swacmod.swacmod test')
    print

###############################################################################
class EndToEndTests(unittest.TestCase):
    """Test suite for the SWAcMod project."""

    specs_file = u.CONSTANTS['SPECS_FILE']
    input_file = u.CONSTANTS['TEST_INPUT_FILE']
    input_dir = u.CONSTANTS['TEST_INPUT_DIR']

    data = io.load_and_validate(specs_file, input_file, input_dir)
    if not data:
        print 'Loading failed, interrupting tests now.'
        sys.exit()

    ids = range(1, data['params']['num_nodes'] + 1)

    def test_keys(self):
        """Test for validate_all() function."""
        all_keys = self.data['series'].keys() + self.data['params'].keys()
        for key in all_keys:
            if key in ['date', 'months', 'kc_list', 'ror_prop', 'ror_limit',
                       'macro_prop', 'macro_limit', 'macro_act', 'macro_rec',
                       'ror_act']:
                continue
            self.assertTrue(key in self.data['specs'])

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

    def test_validate_functions(self):
        """Test that all parameters and series have a validation function."""
        funcs = [i.replace('val_', '') for i in dir(v) if i.startswith('val_')]
        params = io.load_yaml(u.CONSTANTS['SPECS_FILE']).keys()
        self.assertEqual(set(params), set(funcs))

    def test_get_output(self):
        """Test for get_output() function."""
        for node in self.ids:
            output = swacmod.get_output(self.data, node)
            results = io.load_results()
            for key in u.CONSTANTS['COL_ORDER']:
                if key in ['', 'date']:
                    continue
                self.assertTrue(key in results)
                self.assertTrue(key in self.data['series'] or
                                key in output)
                self.assertEqual(len(results) - 1,
                                 len(output))
                if key in self.data['series'] and key not in \
                        ['rainfall_ts', 'pe_ts']:
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
                        print '\n'
                        for col in u.CONSTANTS['COL_ORDER']:
                            if col in ['', 'date']:
                                continue
                            if round(output[col][num] * 1e5) != \
                                    round(results[col][num] * 1e5):
                                print '%s: %s (%s)' % (col, output[col][num],
                                                       results[col][num])
                        print '\n---> Failed at "%s", row %d\n' % (key, num)
                        raise AssertionError(err)


###############################################################################
if __name__ == '__main__':

    unittest.main()
