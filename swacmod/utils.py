#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod utils."""

# Standard Library
import os
import logging

# Third Party Libraries
import psutil

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
    'net_pefac', 'precip_to_ground', 'snowfall_o', 'rainfall_o', 'snowpack',
    'snowmelt', 'net_rainfall', 'rapid_runoff_c', 'rapid_runoff',
    'runoff_recharge', 'macropore', 'percol_in_root', 'rawrew',
    'tawrew', 'p_smd', 'smd', 'k_slope', 'ae', 'unutilised_pe',
    'perc_through_root', 'subroot_leak', 'interflow_bypass',
    'interflow_store_input', 'interflow_volume', 'infiltration_recharge',
    'interflow_to_rivers', 'recharge_store_input', 'recharge_store',
    'combined_recharge', 'combined_str', 'combined_ae', 'evt', 'average_in',
    'average_out', 'balance'
]


###############################################################################
class ValidationError(Exception):
    """General exception for validation errors."""

    pass


###############################################################################
def get_ram_usage_for_process(pid=None):
    """Get memory usage for process given its id.

    If none is given, get memory usage of current process.
    Returns a float (Mb).
    """
    if not pid:
        pid = os.getpid()
    process = psutil.Process(pid)
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


###############################################################################
def weighted_sum(to_sum, weights):
    """Get the weighted sum for a list and its weights."""
    if len(to_sum) != len(weights):
        logging.error('Could not complete weighted sum, different lengths')
        return
    temp = [to_sum[i] * weights[i] for i in range(len(to_sum))]
    return sum(temp)
