#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod utils."""

# Standard Library
import os
import logging

# Third Party Libraries
import psutil
import numpy as np

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

# Header, Column needs conversion, Column needs aggregation
CONSTANTS['BALANCE_CONVERSIONS'] = [
    ('DATE', False, False),
    ('nDays', False, False),
    ('Area', False, True),
    ('Precipitation', True, True),
    ('PEt', True, True),
    ('VegitationPE_PEfac', True, True),
    ('CanopyStorage', True, True),
    ('PEfacLessCanopy_AE', True, True),
    ('Precipitation_Groundlevel', True, True),
    ('Snowfall', True, True),
    ('Precipitation_Rainfall', True, True),
    ('SnowPack', True, True),
    ('SnowMelt', True, True),
    ('Rainfall_SnowMelt', True, True),
    ('RapidRunoff', True, True),
    ('RunoffRecharge', True, True),
    ('MacroPore', True, True),
    ('Percolation_RootZone', True, True),
    ('RAWREW', True, True),
    ('TAWTEW', True, True),
    ('pSMD', True, True),
    ('SMD', True, True),
    ('AE', True, True),
    ('PercolationThroughRootZone', True, True),
    ('SubRootZoneLeakege', True, True),
    ('BypassingInterflow', True, True),
    ('InputInterflowStore', True, True),
    ('InterflowStoreVol', True, True),
    ('InfiltrationRecharge', True, True),
    ('InterflowtoSW', True, True),
    ('InputRechargeStore', True, True),
    ('RechargeStoreVol', True, True),
    ('CombinedRecharge', True, True),
    ('CombinedSW', True, True),
    ('CombinedAE', True, True),
    ('UnitilisedPE', True, True),
    ('AVERAGE_IN', True, True),
    ('AVERAGE_OUT', True, True),
    ('BALANCE', True, True)
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
    """Get the weighted sum given a list and its weights."""
    if len(to_sum) != len(weights):
        logging.error('Could not complete weighted sum, different lengths')
        return
    temp = [to_sum[i] * weights[i] for i in range(len(to_sum))]
    return sum(temp)


###############################################################################
def aggregate_output(data, output, method='sum'):
    """Aggregate all columns according to user-defined time periods."""
    final = {}
    for col in CONSTANTS['COL_ORDER']:
        new_col = aggregate_output_col(data, output, col, method=method)
        final[col] = new_col
    return final


###############################################################################
def aggregate_output_col(data, output, column, method='sum'):
    """Aggregate an output column according to user-defined time periods."""
    times = data['params']['time_periods']
    final = [0.0 for _ in range(len(times))]

    for num, time in enumerate(times):
        if column == 'date':
            date = data['series']['date'][time[1]-1]
            final[num] = date.strftime('%d/%m/%Y')
        else:
            final[num] = np.sum(output[column][time[0]-1:time[1]])
        if method == 'average':
            final[num] /= (time[1] - time[0] + 1)

    return final
