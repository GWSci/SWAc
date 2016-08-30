#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod utils."""

# Standard Library
import os
import re
import logging

CONSTANTS = {}

CONSTANTS['CODE_DIR'] = os.path.dirname(os.path.abspath(__file__))
CONSTANTS['ROOT_DIR'] = os.path.join(CONSTANTS['CODE_DIR'], '../')
CONSTANTS['INPUT_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'input_files')
CONSTANTS['OUTPUT_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'output_files')
CONSTANTS['INPUT_FILE'] = os.path.join(CONSTANTS['INPUT_DIR'], 'input.yml')
CONSTANTS['SPECS_FILE'] = os.path.join(CONSTANTS['CODE_DIR'], 'specs.yml')
CONSTANTS['TEST_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'tests')
CONSTANTS['TEST_INPUT_DIR'] = os.path.join(CONSTANTS['TEST_DIR'],
                                           'input_files')
CONSTANTS['TEST_INPUT_FILE'] = os.path.join(CONSTANTS['TEST_INPUT_DIR'],
                                            'input.yml')
CONSTANTS['TEST_RESULTS_FILE'] = os.path.join(CONSTANTS['TEST_DIR'],
                                              'results.csv')

CONSTANTS['COL_ORDER'] = [
    'date', 'rainfall_ts', 'pe_ts', 'pefac', 'canopy_storage',
    'veg_diff', 'precipitation', 'snowfall_o', 'rainfall_o', 'snowpack',
    'snowmelt', 'net_rainfall', 'rapid_runoff_c', 'rapid_runoff',
    'runoff_recharge', 'macropore', 'perc_in_root', 'rawrew',
    'tawrew', 'p_smd', 'smd', 'k_s', 'ae', 'unutilized_pe',
    'perc_through_root', 'subroot_leak', 'interflow_bypass',
    'interflow_input', 'interflow_volume', 'infiltration_recharge',
    'interflow_to_rivers', 'recharge_input', 'recharge_store',
    'combined_recharge', 'str', 'combined_ae', 'evt', 'average_in',
    'average_out', 'balance'
]


###############################################################################
class ValidationError(Exception):
    """General exception for validation errors."""

    pass


###############################################################################
def normalize_default_value(data, param):
    """Normalize default value for a parameter."""
    default = data['specs'][param]['default']
    pattern = re.findall(r'\{.*?\}', default)
    if pattern:
        new_value = re.sub(r'[\{\}]', '', pattern[0])
        data['params'][param] = re.sub(pattern[0], data['params'][new_value],
                                       default)
        logging.info('\t\tDefaulted "%s" to %s', param, data['params'][param])


###############################################################################
def weighted_sum(to_sum, weights):
    """Get the weighted sum for a list and its weights."""
    if len(to_sum) != len(weights):
        logging.error('Could not complete weighted sum, different lengths')
        return
    temp = [to_sum[i] * weights[i] for i in range(len(to_sum))]
    return sum(temp)
