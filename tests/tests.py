#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod tests."""

# Standard Library
import unittest

# Internal modules
from swacmod import io
from swacmod import swacmod
from swacmod import utils as u
from swacmod import validation as v


###############################################################################
class EndToEndTests(unittest.TestCase):
    """Test suite for the SWAcMod project."""

    data = {}
    input_file = u.CONSTANTS['TEST_INPUT_FILE']
    input_dir = u.CONSTANTS['TEST_INPUT_DIR']
    specs, series, params = io.load_params_from_yaml(input_file=input_file,
                                                     input_dir=input_dir)
    data['specs'], data['series'], data['params'] = specs, series, params

    ids = range(1, data['params']['num_nodes'] + 1)
    data['output'] = dict((k, {}) for k in ids)

    def test_validate_all(self):
        """Test for validate_all() function."""
        for key in self.data['series'].keys() + self.data['params'].keys():
            if key in ['date', 'TAW', 'RAW']:
                continue
            self.assertTrue(key in self.data['specs'])
        self.assertEqual(len(self.data['specs']), 44)
        io.validate_all(self.data)

    def test_validate_functions(self):
        """Test that all parameters and series have a validation function."""
        funcs = [i.replace('val_', '') for i in dir(v) if i.startswith('val_')]
        params = io.yaml.load(open(u.CONSTANTS['SPECS_FILE'], 'r')).keys()
        self.assertEqual(set(params), set(funcs))

    def test_get_output(self):
        """Test for get_output() function."""
        for node in self.ids:
            swacmod.get_output(self.data, node)
            results = io.load_results()
            for key in u.CONSTANTS['COL_ORDER']:
                if key in ['', 'date']:
                    continue
                self.assertTrue(key in results)
                self.assertTrue(key in self.data['series'] or
                                key in self.data['output'][node])
                self.assertEqual(len(results) - 1,
                                 len(self.data['output'][node]) + 2)
                if key in self.data['series']:
                    if isinstance(self.data['series'][key][0], list):
                        new_list = [i[0] for i in self.data['series'][key]]
                    else:
                        new_list = self.data['series'][key]
                else:
                    new_list = self.data['output'][node][key]
                self.assertEqual(len(new_list), len(results[key]))
                for num, item in enumerate(new_list):
                    try:
                        self.assertAlmostEqual(results[key][num],
                                               item,
                                               places=5)
                    except AssertionError as err:
                        print '\n\n---> Failed at "%s", row %d\n' % (key, num)
                        raise AssertionError(err)


###############################################################################
if __name__ == '__main__':

    unittest.main()
