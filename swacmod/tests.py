#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod tests."""

# Standard Library
import unittest

# Internal modules
from . import io
from . import swacmod
from . import utils as u
from . import validation as v


###############################################################################
class EndToEndTests(unittest.TestCase):
    """Test suite for the SWAcMod project."""

    def test_load_params_from_yaml(self):
        """Test for load_params_from_yaml() function."""
        specs = io.yaml.load(open(u.CONSTANTS['SPECS_FILE'], 'r'))
        series, params = io.load_params_from_yaml()
        for key in params.keys() + series.keys():
            self.assertTrue(key in specs)
        self.assertEqual(len(specs), 44)

    def test_validate_all(self):
        """Test for validate_all() function."""
        data = {'specs': io.yaml.load(open(u.CONSTANTS['SPECS_FILE'], 'r'))}
        data['series'], data['params'] = io.load_params_from_yaml()
        io.validate_all(data)

    def test_validate_functions(self):
        """Test that all parameters and series have a validation function."""
        funcs = [i.replace('val_', '') for i in dir(v) if i.startswith('val_')]
        params = io.yaml.load(open(u.CONSTANTS['SPECS_FILE'], 'r')).keys()
        self.assertEqual(set(params), set(funcs))

    def test_load_input_from_excel(self):
        """Test for load_input_from_excel() function."""
        series = io.load_input_from_excel()
        self.assertTrue('date' in series)
        for key in series:
            self.assertEqual(len(series[key]), len(series['date']))

    def test_load_params_from_excel(self):
        """Test for load_params_from_excel() function."""
        params = io.load_params_from_excel()
        for key in ['recharge_proportion', 'recharge_limit',
                    'macropore_proportion', 'macropore_limit', 'ZR', 'KC']:
            self.assertEqual(len(params[key]), 12)

    def test_oned_get_output(self):
        """Test for get_output() function (1d model)."""
        data = {}
        data['series'] = io.load_input_from_excel()
        data['params'] = io.load_params_from_excel()
        swacmod.get_output(data)
        results = io.load_results()
        for key in u.CONSTANTS['COL_ORDER']:
            if key == '':
                continue
            self.assertTrue(key in results)
            self.assertTrue(key in data['series'] or key in data['output'])
            self.assertEqual(len(results) - 2, len(data['output']) + 2)
            if key in data['series']:
                new_list = data['series'][key]
            else:
                new_list = data['output'][key]
            self.assertEqual(len(new_list), len(results[key]))
            for num, item in enumerate(new_list):
                try:
                    self.assertAlmostEqual(results[key][num],
                                           item,
                                           places=5)
                except AssertionError as err:
                    print '\n---> Failed at "%s", row %d\n' % (key, num)
                    raise AssertionError(err)

    def test_twod_get_output(self):
        """Test for get_output() function (2d model)."""


###############################################################################
if __name__ == '__main__':

    unittest.main()
