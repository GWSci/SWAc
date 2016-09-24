#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod utils."""

# Standard Library
import os
import sys
import logging
import datetime
import subprocess as sp

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
    ('DATE', False),
    ('nDays', False),
    ('Area', False),
    ('Precipitation', True),
    ('PEt', True),
    ('VegitationPE_PEfac', True),
    ('CanopyStorage', True),
    ('PEfacLessCanopy_AE', True),
    ('Precipitation_Groundlevel', True),
    ('Snowfall', True),
    ('Precipitation_Rainfall', True),
    ('SnowPack', True),
    ('SnowMelt', True),
    ('Rainfall_SnowMelt', True),
    ('RapidRunoff', True),
    ('RunoffRecharge', True),
    ('MacroPore', True),
    ('Percolation_RootZone', True),
    ('RAWREW', True),
    ('TAWTEW', True),
    ('pSMD', True),
    ('SMD', True),
    ('AE', True),
    ('PercolationThroughRootZone', True),
    ('SubRootZoneLeakege', True),
    ('BypassingInterflow', True),
    ('InputInterflowStore', True),
    ('InterflowStoreVol', True),
    ('InfiltrationRecharge', True),
    ('InterflowtoSW', True),
    ('InputRechargeStore', True),
    ('RechargeStoreVol', True),
    ('CombinedRecharge', True),
    ('CombinedSW', True),
    ('CombinedAE', True),
    ('UnitilisedPE', True),
    ('AVERAGE_IN', True),
    ('AVERAGE_OUT', True),
    ('BALANCE', True)
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


###############################################################################
def get_modified_time(path):
    """Get the datetime a file was modified."""
    try:
        mod = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        mod = datetime.datetime(mod.year, mod.month, mod.day, mod.hour,
                                mod.minute, mod.second)
    except OSError:
        logging.error('Could not find %s, set modified time to 1/1/1901', path)
        mod = datetime.datetime(1901, 1, 1, 0, 0, 0)
    return mod


###############################################################################
def compile_model():
    """Compile Cython model."""
    mod_c = get_modified_time('swacmod/model.c')
    mod_pyx = get_modified_time('swacmod/model.pyx')
    if mod_pyx >= mod_c:
        print 'model.pyx modified, recompiling'
        proc = sp.Popen(['python', 'setup.py', 'build_ext', '--inplace'],
                        cwd=CONSTANTS['CODE_DIR'],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
        if proc.returncode != 0:
            print 'Could not compile C extensions:'
            print '%s' % proc.stdout.read()
            print '%s' % proc.stderr.read()
            sys.exit(proc.returncode)
