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
    'runoff_recharge', 'macropore_att', 'macropore_dir', 'percol_in_root',
    'rawrew', 'tawtew', 'p_smd', 'smd', 'k_slope', 'ae', 'unutilised_pe',
    'rejected_recharge', 'perc_through_root', 'subroot_leak',
    'interflow_bypass', 'interflow_store_input', 'interflow_volume',
    'infiltration_recharge', 'interflow_to_rivers', 'recharge_store_input',
    'recharge_store', 'combined_recharge', 'sw_attenuation', 'combined_str',
    'combined_ae', 'evt', 'average_in', 'average_out', 'total_storage_change',
    'balance'
]

# Header
# Column needs conversion
# Column included in reduced output
CONSTANTS['BALANCE_CONVERSIONS'] = [
    ('DATE', False, True),
    ('nDays', False, True),
    ('Area', False, True),
    ('Precipitation', True, False),
    ('PEt', True, False),
    ('VegetationPE_PEfac', True, False),
    ('CanopyStorage', True, False),
    ('PEfacLessCanopy_AE', True, False),
    ('Precipitation_Groundlevel', True, False),
    ('Snowfall', True, False),
    ('Precipitation_Rainfall', True, False),
    ('SnowPack', True, False),
    ('SnowMelt', True, False),
    ('Rainfall_SnowMelt', True, False),
    ('RapidRunoff', True, False),
    ('RunoffRecharge', True, False),
    ('MacroPoreAttenuated', True, False),
    ('MacroPoreDirect', True, False),
    ('Percolation_RootZone', True, False),
    ('RAWREW', True, False),
    ('TAWTEW', True, False),
    ('pSMD', True, False),
    ('SMD', True, False),
    ('AE', True, False),
    ('RejectedRecharge', True, False),
    ('PercolationThroughRootZone', True, False),
    ('SubRootZoneLeakege', True, False),
    ('BypassingInterflow', True, False),
    ('InputInterflowStore', True, False),
    ('InterflowStoreVol', True, False),
    ('InfiltrationRecharge', True, False),
    ('InterflowtoSW', True, False),
    ('InputRechargeStore', True, False),
    ('RechargeStoreVol', True, False),
    ('CombinedRecharge', True, True),
    ('SWAttenuation', True, False),
    ('CombinedSW', True, True),
    ('CombinedAE', True, True),
    ('UnitilisedPE', True, True),
    ('AVERAGE_IN', True, False),
    ('AVERAGE_OUT', True, False),
    ('TOTAL_STORAGE_CHANGE', True, False),
    ('BALANCE', True, False)
]


###############################################################################
class ValidationError(Exception):
    """General exception for validation errors."""

    def __init__(self, msg):
        """Initialization."""
        new_msg = '---> Validation failed: %s' % msg
        Exception.__init__(self, new_msg)


###############################################################################
class FinalizationError(Exception):
    """General exception for validation errors."""

    def __init__(self, msg):
        """Initialization."""
        new_msg = '---> Finalization failed: %s' % msg
        Exception.__init__(self, new_msg)


###############################################################################
class InputOutputError(Exception):
    """General exception for validation errors."""

    def __init__(self, msg):
        """Initialization."""
        new_msg = '---> InputOutput failed: %s' % msg
        Exception.__init__(self, new_msg)


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
            date = data['series']['date'][time[1]-2]
            final[num] = date.strftime('%d/%m/%Y')
        else:
            final[num] = np.sum(output[column][time[0]-1:time[1]-1])
        if method == 'average':
            final[num] /= (time[1] - time[0])

    return final


###############################################################################
def get_modified_time(path):
    """Get the datetime a file was modified."""
    try:
        mod = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        mod = datetime.datetime(mod.year, mod.month, mod.day, mod.hour,
                                mod.minute, mod.second)
    except OSError:
        logging.warning('Could not find %s, set modified time to 1/1/1901',
                        path)
        mod = datetime.datetime(1901, 1, 1, 0, 0, 0)
    return mod


###############################################################################
def build_taw_raw(params):
    """Build the TAW and RAW matrices."""
    taw, raw = {}, {}

    for node in range(1, params['num_nodes'] + 1):
        taw[node], raw[node] = [], []
        fcp = params['soil_static_params']['FC']
        wpp = params['soil_static_params']['WP']
        ppp = params['soil_static_params']['p']
        pss = params['soil_spatial'][node]
        lus = params['lu_spatial'][node]
        var1 = [(fcp[i] - wpp[i]) * pss[i] * 1000 for i in range(len(pss))]
        var2 = [ppp[i] * pss[i] for i in range(len(pss))]
        for num in range(12):
            var3 = [params['zr'][num+1][i] * lus[i] for i in range(len(lus))]
            taw[node].append(sum(var1) * sum(var3))
            raw = taw[node][num] * sum(var2)
            raw[node].append(raw)

    return taw, raw


###############################################################################
def invert_taw_raw(param, params):
    """Invert the TAW and RAW matrices, from month by zone to node by month."""
    new_param = {}

    for node in range(1, params['num_nodes'] + 1):
        new_param[node] = []
        lus = params['lu_spatial'][node]
        for num in range(1, 13):
            value = [param[num][i] * lus[i] for i in range(len(lus))]
            new_param[node].append(sum(value))

    return new_param


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
