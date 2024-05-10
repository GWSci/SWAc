#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod utils."""

# Standard Library
import os
import sys
import struct
import logging
import datetime
import subprocess as sp

# Third Party Libraries
import psutil
import numpy as np
from calendar import monthrange

import swacmod.feature_flags as ff

CONSTANTS = {}

CONSTANTS['CODE_DIR'] = os.path.dirname(os.path.abspath(__file__))
CONSTANTS['ROOT_DIR'] = os.path.join(CONSTANTS['CODE_DIR'], '../')
CONSTANTS['INPUT_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'input_files')
CONSTANTS['OUTPUT_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'output_files')
CONSTANTS['INPUT_FILE'] = os.path.join(CONSTANTS['INPUT_DIR'], 'input.yml')
CONSTANTS['SPECS_FILE'] = os.path.join(CONSTANTS['CODE_DIR'], 'specs.yml')

CONSTANTS['TEST_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'test/tests')
CONSTANTS['TEST_INPUT_DIR'] = os.path.join(CONSTANTS['TEST_DIR'],
                                           'input_files')
CONSTANTS['TEST_INPUT_FILE'] = os.path.join(CONSTANTS['TEST_INPUT_DIR'],
                                            'input.yml')
CONSTANTS['TEST_RESULTS_FILE'] = os.path.join(CONSTANTS['TEST_DIR'],
                                              'results.csv')

def col_order():
    if ff.use_natproc:
        return [
            'date', 'rainfall_ts', 'pe_ts', 'pefac', 'canopy_storage', 'net_pefac',
            'precip_to_ground', 'snowfall_o', 'rainfall_o', 'snowpack', 'snowmelt',
            'net_rainfall', 'rapid_runoff', 'runoff_recharge', 'macropore_att',
            'macropore_dir', 'percol_in_root', 'rawrew', 'tawtew', 'p_smd', 'smd',
            'ae', 'rejected_recharge', 'perc_through_root', 'subroot_leak',
            'interflow_bypass', 'interflow_store_input', 'interflow_volume',
            'infiltration_recharge', 'interflow_to_rivers', 'recharge_store_input',
            'recharge_store', 'combined_recharge', 'atten_input',
            'sw_attenuation', 'pond_direct', 'pond_atten', 'open_water_ae',
            'atten_input_actual',
            'pond_over', 'sw_other', 'open_water_evap', 'swabs_ts',
            'swdis_ts', 'combined_str', 'combined_ae', 'evt', 'average_in',
            'average_out', 'total_storage_change', 'balance'
        ]
    else:
        return [
            'date', 'rainfall_ts', 'pe_ts', 'pefac', 'canopy_storage', 'net_pefac',
            'precip_to_ground', 'snowfall_o', 'rainfall_o', 'snowpack', 'snowmelt',
            'net_rainfall', 'rapid_runoff', 'runoff_recharge', 'macropore_att',
            'macropore_dir', 'percol_in_root', 'rawrew', 'tawtew', 'p_smd', 'smd',
            'ae', 'rejected_recharge', 'perc_through_root', 'subroot_leak',
            'interflow_bypass', 'interflow_store_input', 'interflow_volume',
            'infiltration_recharge', 'interflow_to_rivers', 'recharge_store_input',
            'recharge_store', 'combined_recharge', 'sw_attenuation', 'swabs_ts',
            'swdis_ts', 'combined_str', 'combined_ae', 'evt', 'average_in',
            'average_out', 'total_storage_change', 'balance'
        ]

if ff.use_natproc:
    CONSTANTS['COL_ORDER'] = [
        'date', 'rainfall_ts', 'pe_ts', 'pefac', 'canopy_storage', 'net_pefac',
        'precip_to_ground', 'snowfall_o', 'rainfall_o', 'snowpack', 'snowmelt',
        'net_rainfall', 'rapid_runoff', 'runoff_recharge', 'macropore_att',
        'macropore_dir', 'percol_in_root', 'rawrew', 'tawtew', 'p_smd', 'smd',
        'ae', 'rejected_recharge', 'perc_through_root', 'subroot_leak',
        'interflow_bypass', 'interflow_store_input', 'interflow_volume',
        'infiltration_recharge', 'interflow_to_rivers', 'recharge_store_input',
        'recharge_store', 'combined_recharge', 'atten_input',
        'sw_attenuation', 'pond_direct', 'pond_atten', 'open_water_ae',
        'atten_input_actual',
        'pond_over', 'sw_other', 'open_water_evap', 'swabs_ts',
        'swdis_ts', 'combined_str', 'combined_ae', 'evt', 'average_in',
        'average_out', 'total_storage_change', 'balance'
    ]
else:
    CONSTANTS['COL_ORDER'] = [
        'date', 'rainfall_ts', 'pe_ts', 'pefac', 'canopy_storage', 'net_pefac',
        'precip_to_ground', 'snowfall_o', 'rainfall_o', 'snowpack', 'snowmelt',
        'net_rainfall', 'rapid_runoff', 'runoff_recharge', 'macropore_att',
        'macropore_dir', 'percol_in_root', 'rawrew', 'tawtew', 'p_smd', 'smd',
        'ae', 'rejected_recharge', 'perc_through_root', 'subroot_leak',
        'interflow_bypass', 'interflow_store_input', 'interflow_volume',
        'infiltration_recharge', 'interflow_to_rivers', 'recharge_store_input',
        'recharge_store', 'combined_recharge', 'sw_attenuation', 'swabs_ts',
        'swdis_ts', 'combined_str', 'combined_ae', 'evt', 'average_in',
        'average_out', 'total_storage_change', 'balance'
    ]

def full_area(area, ponded_fraction):
    return np.float64(area)

def ponded_area(area, ponded_fraction):
    return np.float64(area * ponded_fraction)

def not_ponded_area(area, ponded_fraction):
    return np.float64(area * (1.0 - ponded_fraction))

def area_fn():
    # populate area_fn with default area
    result = {p: full_area for p in col_order()}

    # not in list above
    result['k_slope'] = full_area
    result['historical_nitrate_reaching_water_table_array_tons_per_day'] = full_area
    result['nitrate_reaching_water_table_array_tons_per_day'] = full_area
    result['nitrate_to_surface_water_array_tons_per_day'] = full_area
    result['mi_array_kg_per_day'] = full_area
    result['rapid_runoff_c'] = not_ponded_area

    for p in ['canopy_storage', 'precip_to_ground', 'rapid_runoff', 'runoff_recharge',
            'macropore_att', 'macropore_dir', 'percol_in_root', 'p_smd', 'smd', 'ae',
            'rejected_recharge', 'perc_through_root', 'interflow_bypass',
            'interflow_store_input', 'interflow_volume', 'infiltration_recharge',
            'interflow_to_rivers', 'unutilised_pe']:
        result[p] = not_ponded_area

    for p in ['sw_attenuation', 'pond_direct', 'pond_atten', 'pond_over', 'sw_other']:
        result[p] = ponded_area
    return result

# Header
# Column needs conversion
# Column included in reduced output
def balance_conversions():
    if ff.use_natproc:
        return [
            ('DATE', False, True), ('nDays', False, True), ('Area', False, True),
            ('Precipitation', True, False), ('PEt', True, False),
            ('VegetationPE_PEfac', True, False), ('CanopyStorage', True, False),
            ('PEfacLessCanopy_AE', True, False),
            ('Precipitation_Groundlevel', True, False), ('Snowfall', True, False),
            ('Precipitation_Rainfall', True, False), ('SnowPack', True, False),
            ('SnowMelt', True, False), ('Rainfall_SnowMelt', True, False),
            ('RapidRunoff', True, False), ('RunoffRecharge', True, False),
            ('MacroPoreAttenuated', True, False), ('MacroPoreDirect', True, False),
            ('Percolation_RootZone', True, False), ('RAWREW', True, False),
            ('TAWTEW', True, False), ('pSMD', True, False), ('SMD', True, False),
            ('AE', True, False), ('RejectedRecharge', True, False),
            ('PercolationThroughRootZone', True, False),
            ('SubRootZoneLeakege', True, False), ('BypassingInterflow', True, False),
            ('InputInterflowStore', True, False), ('InterflowStoreVol', True, False),
            ('InfiltrationRecharge', True, False), ('InterflowtoSW', True, False),
            ('InputRechargeStore', True, False), ('RechargeStoreVol', True, False),
            ('CombinedRecharge', True, True), ('SWAttenInputPot', True, False),
            ('SWAttenuation', True, False),
            ('PondDirect', True, False),('PondAtten', True, False),
            ('OpenWaterAE', True, False),('SWAttenInputAct', True, False),
            ('PondOverspill', True, False), ('OtherSW', True, False),
            ('OpenWaterPE', True, False),
            ('SWAbstractions', False, False), ('SWDischarges', False, False),
            ('CombinedSW', True, True), ('CombinedAE', True, True),
            ('UnitilisedPE', True, True), ('AVERAGE_IN', True, False),
            ('AVERAGE_OUT', True, False), ('TOTAL_STORAGE_CHANGE', True, False),
            ('BALANCE', True, False)
        ]
    else:
        return [
            ('DATE', False, True), ('nDays', False, True), ('Area', False, True),
            ('Precipitation', True, False), ('PEt', True, False),
            ('VegetationPE_PEfac', True, False), ('CanopyStorage', True, False),
            ('PEfacLessCanopy_AE', True, False),
            ('Precipitation_Groundlevel', True, False), ('Snowfall', True, False),
            ('Precipitation_Rainfall', True, False), ('SnowPack', True, False),
            ('SnowMelt', True, False), ('Rainfall_SnowMelt', True, False),
            ('RapidRunoff', True, False), ('RunoffRecharge', True, False),
            ('MacroPoreAttenuated', True, False), ('MacroPoreDirect', True, False),
            ('Percolation_RootZone', True, False), ('RAWREW', True, False),
            ('TAWTEW', True, False), ('pSMD', True, False), ('SMD', True, False),
            ('AE', True, False), ('RejectedRecharge', True, False),
            ('PercolationThroughRootZone', True, False),
            ('SubRootZoneLeakege', True, False), ('BypassingInterflow', True, False),
            ('InputInterflowStore', True, False), ('InterflowStoreVol', True, False),
            ('InfiltrationRecharge', True, False), ('InterflowtoSW', True, False),
            ('InputRechargeStore', True, False), ('RechargeStoreVol', True, False),
            ('CombinedRecharge', True, True), ('SWAttenuation', True, False),
            ('SWAbstractions', False, False), ('SWDischarges', False, False),
            ('CombinedSW', True, True), ('CombinedAE', True, True),
            ('UnitilisedPE', True, True), ('AVERAGE_IN', True, False),
            ('AVERAGE_OUT', True, False), ('TOTAL_STORAGE_CHANGE', True, False),
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
    mem = process.memory_info()[0] / float(2**20)
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
    for col in col_order():
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
            date = data['series']['date'][time[1] - 2]
            final[num] = date.strftime('%d/%m/%Y')
        else:
            final[num] = np.sum(output[column][time[0] - 1:time[1] - 1])
        if method == 'average':
            final[num] /= (time[1] - time[0])
            # if column == 'recharge':
            #     print final[num]
    return final


###############################################################################
def aggregate_array(data, array, method='average'):
    """Aggregate 1d array according to user-defined time periods."""
    times = data['params']['time_periods']
    final = np.zeros((len(times)), dtype=np.float64)

    for num, time in enumerate(times):
        final[num] = np.sum(array[time[0] - 1:time[1] - 1])
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
            var3 = [params['zr'][num + 1][i] * lus[i] for i in range(len(lus))]
            taw[node].append(sum(var1) * sum(var3))
            raw = taw[node][num] * sum(var2)
            raw[node].append(raw) # As far as I can tell, this line will always result in an error. I don't think this code has ever been run.

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
    mod_c = get_modified_time(os.path.join(CONSTANTS['CODE_DIR'], 'cymodel.c'))
    mod_pyx = get_modified_time(
        os.path.join(CONSTANTS['CODE_DIR'], 'cymodel.pyx'))
    if mod_pyx >= mod_c:
        arch = struct.calcsize('P') * 8
        print('cymodel.pyx modified, recompiling for %d-bit' % arch)
        proc = sp.Popen([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                        cwd=CONSTANTS['CODE_DIR'],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
        if proc.returncode != 0:
            print('Could not compile C extensions:')
            print('%s' % proc.stdout.read())
            print('%s' % proc.stderr.read())
            sys.exit(proc.returncode)
        boo = True
    else:
        boo = False
    return boo

###############################################################################


def monthdelta(d1, d2):
    " difference in months between two dates"

    from calendar import monthrange

    delta = 0
    while True:
        mdays = monthrange(d1.year, d1.month)[1]
        d1 += datetime.timedelta(days=mdays)
        if d1 <= d2:
            delta += 1
        else:
            break
    return delta

def monthdelta2(d1, d2):
    " difference in months between two dates"
    year_diff = d2.year - d1.year
    month_diff = d2.month - d1.month
    day_correction = (-1) if d2.day < d1.day else 0
    result = (12 * year_diff) + month_diff + day_correction
    return result


def weekdelta(d1, d2):
    " difference in weeks between two dates"

    monday1 = (d1 - datetime.timedelta(days=d1.weekday()))
    monday2 = (d2 - datetime.timedelta(days=d2.weekday()))

    return (monday2 - monday1).days / 7


def daterange(start_date, end_date):
    "iterate over dates"
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def month_indices(month, data):
    "get day numbers of a month"
    last = datetime.timedelta(data['params']['time_periods'][-1][1]-1)
    first = data["params"]["start_date"]

    return [
        i for i, day_date in enumerate(daterange(first, first+last))
        if day_date.month == month
    ]
