#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod validation functions."""

# Standard Library
import re
import logging
import datetime

# Third Party Libraries
import numpy as np

# Internal modules
from . import utils as u


###############################################################################
def fin_start_date(data, name):
    """Finalize the "start_date" parameter i/o."""
    params = data['params']

    new_date = str(params['start_date'])
    fields = re.findall(r'^(\d{4})-(\d{2})-(\d{2})$', new_date)
    if not fields:
        msg = ('start_date has to be in the format YYYY-MM-DD '
               '(e.g. 1980-01-13)')
        raise u.ValidationError(msg)
    params['start_date'] = datetime.datetime(int(fields[0][0]),
                                             int(fields[0][1]),
                                             int(fields[0][2]))


###############################################################################
def fin_date(data, name):
    """Finalize the "date" series i/o."""
    series, params = data['series'], data['params']

    max_time = max([i for j in params['time_periods'] for i in j])
    day = datetime.timedelta(1)
    series['date'] = [params['start_date'] + day * num for num in
                      range(max_time)]
    dates = np.array([np.datetime64(str(i.date())) for i in series['date']])
    series['months'] = dates.astype('datetime64[M]').astype(int) % 12


###############################################################################
def fin_output_individual(data, name):
    """Finalize the "output_individual" parameter i/o."""
    params = data['params']

    oip = str(params['output_individual']).lower()
    sections = [i.strip() for i in oip.split(',')]
    final = []
    for section in sections:
        if section == 'all':
            final = range(1, params['num_nodes'] + 1)
            break
        elif section == 'none':
            final = []
            break
        if '-' in section:
            try:
                first = int(section.split('-')[0].strip())
                second = int(section.split('-')[1].strip())
                final += range(first, second + 1)
            except (TypeError, ValueError):
                pass
        else:
            try:
                final.append(int(section))
            except (TypeError, ValueError):
                pass

    params['output_individual'] = set(final)


###############################################################################
def fin_taw_and_raw(data, name):
    """Finalize the "TAW" and "RAW" parameters i/o."""
    params = data['params']

    params['TAW'], params['RAW'] = {}, {}

    for node in range(1, params['num_nodes'] + 1):
        params['TAW'][node], params['RAW'][node] = [], []
        fcp = params['soil_static_params']['FC']
        wpp = params['soil_static_params']['WP']
        ppp = params['soil_static_params']['p']
        pss = params['soil_spatial'][node]
        lus = params['lu_spatial'][node]
        var1 = [(fcp[i] - wpp[i]) * pss[i] * 1000 for i in range(len(pss))]
        var2 = [ppp[i] * pss[i] for i in range(len(pss))]
        for num in range(12):
            var3 = [params['zr'][num+1][i] * lus[i] for i in range(len(lus))]
            params['TAW'][node].append(sum(var1) * sum(var3))
            params['RAW'][node].append(params['TAW'][node][num] * sum(var2))
        params['TAW'][node] = np.array(params['TAW'][node])
        params['RAW'][node] = np.array(params['RAW'][node])


###############################################################################
def fin_kc_list(data, name):
    """Convert dictionaries to numpy arrays for efficiency."""
    params = data['params']

    params['kc_list'] = sorted(params['kc'].items(), key=lambda x: x[0])
    params['kc_list'] = np.array([i[1] for i in params['kc_list']])


###############################################################################
def fin_ror_prop(data, name):
    """Convert dictionaries to numpy arrays for efficiency."""
    params = data['params']

    params['ror_prop'] = sorted(params['rorecharge_proportion'].items(),
                                key=lambda x: x[0])
    params['ror_prop'] = np.array([i[1] for i in params['ror_prop']])


###############################################################################
def fin_ror_limit(data, name):
    """Convert dictionaries to numpy arrays for efficiency."""
    params = data['params']

    params['ror_limit'] = sorted(params['rorecharge_limit'].items(),
                                 key=lambda x: x[0])
    params['ror_limit'] = np.array([i[1] for i in params['ror_limit']])


###############################################################################
def fin_macro_prop(data, name):
    """Convert dictionaries to numpy arrays for efficiency."""
    params = data['params']

    params['macro_prop'] = sorted(params['macropore_proportion'].items(),
                                  key=lambda x: x[0])
    params['macro_prop'] = np.array([i[1] for i in params['macro_prop']])


###############################################################################
def fin_macro_limit(data, name):
    """Convert dictionaries to numpy arrays for efficiency."""
    params = data['params']

    params['macro_limit'] = sorted(params['macropore_limit'].items(),
                                   key=lambda x: x[0])
    params['macro_limit'] = np.array([i[1] for i in params['macro_limit']])


###############################################################################
def fin_pe_ts(data, name):
    """Finalize the "pe_ts" series i/o."""
    series, specs, params = data['series'], data['specs'], data['params']

    fao = params['fao_process']
    canopy = params['canopy_process']
    if fao != 'enabled' and canopy != 'enabled':
        series[name] = np.zeros(len(series['date']))
        logging.info('\t\tDefaulted "%s" to 0.0', series)
    else:
        specs[name]['required'] = True
        logging.info('\t\tSwitched "%s" to "required"', name)


###############################################################################
def fin_temperature_ts(data, name):
    """Finalize the "temperature_ts" series i/o."""
    specs, params = data['specs'], data['params']

    if params['snow_process'] == 'enabled':
        specs[name]['required'] = True
        logging.info('\t\tSwitched "%s" to "required"', name)


###############################################################################
def fin_subroot_leakage_ts(data, name):
    """Finalize the "subroot_leakage_ts" series i/o."""
    specs, params = data['specs'], data['params']

    if params['leakage_process'] == 'enabled':
        specs[name]['required'] = True
        logging.info('\t\tSwitched "%s" to "required"', name)


###############################################################################
def finalize_params(data):
    """Finalize all parameters."""
    logging.info('\tFinalizing parameters')

    for function in [fin_start_date,
                     fin_date,
                     fin_output_individual,
                     fin_taw_and_raw,
                     fin_kc_list,
                     fin_ror_prop,
                     fin_ror_limit,
                     fin_macro_prop,
                     fin_macro_limit]:

        param = function.__name__.replace('fin_', '')
        function(data, param)
        logging.debug('\t\t"%s" finalized', param)

    logging.info('\tDone.')


###############################################################################
def finalize_series(data):
    """Finalize all time series."""
    logging.info('\tFinalizing time series')

    for function in [fin_pe_ts,
                     fin_temperature_ts,
                     fin_subroot_leakage_ts]:

        series = function.__name__.replace('fin_', '')
        function(data, series)
        logging.debug('\t\t"%s" finalized', series)

    logging.info('\tDone.')
