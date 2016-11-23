#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod validation functions."""

# Standard Library
import re
import logging
import datetime
import multiprocessing

# Third Party Libraries
import numpy as np

# Internal modules
from . import utils as u


###############################################################################
def fin_start_date(data, name):
    """Finalize the "start_date" parameter.

    1) if in the right format, convert it to datetime object.
    """
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
def fin_run_name(data, name):
    """Finalize the "run_name" parameter.

    1) if not a string, convert it.
    2) replace non-alphanumeric characters with underscores.
    """
    params = data['params']
    rnm = params[name]

    if not isinstance(rnm, basestring):
        params[name] = str(rnm)
        logging.info('\t\tConverted "%s" to string', name)

    new_value = re.sub(r'[^a-zA-Z\-0-9]', '_', params[name])
    if new_value != data['params'][name]:
        params[name] = new_value
        logging.info('\t\tNew "%s": %s', name, new_value)


###############################################################################
def fin_num_cores(data, name):
    """Finalize the "num_cores" parameter.

    1) if not provided, use the number of cores of the machine.
    """
    if data['params'][name] is None:
        count = multiprocessing.cpu_count()
        data['params'][name] = count
        logging.info('\t\tDefaulted "%s" to %s', name, count)


###############################################################################
def fin_output_recharge(data, name):
    """Finalize the "output_recharge" parameter.

    1) if not provided, set it to True.
    """
    if data['params'][name] is None:
        data['params'][name] = True


###############################################################################
def fin_output_individual(data, name):
    """Finalize the "output_individual" parameter.

    1) if not provided, set it to "none".
    2) convert it to a string if it's not.
    3) parse it into a set of integers.
    """
    params = data['params']

    if params[name] is None:
        params[name] = 'none'

    oip = str(params[name]).lower()
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

    params[name] = set(final)


###############################################################################
def fin_irchcb(data, name):
    """Finalize the "irchcb" parameter.

    1) if not provided, set it to 50.
    """
    if data['params'][name] is None:
        data['params'][name] = 50


###############################################################################
def fin_nodes_per_line(data, name):
    """Finalize the "nodes_per_line" parameter.

    1) if not provided, set it to 10.
    """
    if data['params'][name] is None:
        data['params'][name] = 10


###############################################################################
def fin_reporting_zone_mapping(data, name):
    """Finalize the "reporting_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, 1) for k in range(1, nodes + 1))


###############################################################################
def fin_reporting_zone_names(data, name):
    """Finalize the "reporting_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['reporting_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_rainfall_zone_names(data, name):
    """Finalize the "rainfall_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['rainfall_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_pe_zone_names(data, name):
    """Finalize the "pe_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['pe_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_temperature_zone_mapping(data, name):
    """Finalize the "temperature_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data['params']
    if params[name] is None:
        nodes = params['num_nodes']
        params[name] = dict((k, 1) for k in range(1, nodes + 1))


###############################################################################
def fin_temperature_zone_names(data, name):
    """Finalize the "temperature_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['temperature_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_subroot_zone_mapping(data, name):
    """Finalize the "subroot_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data['params']
    if params[name] is None:
        nodes = params['num_nodes']
        params[name] = dict((k, [1, 1.0]) for k in range(1, nodes + 1))


###############################################################################
def fin_subroot_zone_names(data, name):
    """Finalize the "subroot_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        values = [i[0] for i in params['subroot_zone_mapping'].values()]
        zones = len(set(values))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_rapid_runoff_zone_mapping(data, name):
    """Finalize the "rapid_runoff_zone_mapping" parameter.

    1) if not provided, set it to all 0s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, 0) for k in range(1, nodes + 1))


###############################################################################
def fin_rapid_runoff_zone_names(data, name):
    """Finalize the "rapid_runoff_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['rapid_runoff_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_rorecharge_zone_mapping(data, name):
    """Finalize the "rorecharge_zone_mapping" parameter.

    1) if not provided, set it to all 0s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, 0) for k in range(1, nodes + 1))


###############################################################################
def fin_rorecharge_zone_names(data, name):
    """Finalize the "rorecharge_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['rorecharge_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_macropore_zone_mapping(data, name):
    """Finalize the "macropore_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, 1) for k in range(1, nodes + 1))


###############################################################################
def fin_macropore_zone_names(data, name):
    """Finalize the "macropore_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['macropore_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_soil_zone_names(data, name):
    """Finalize the "macropore_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        try:
            zones = len(params['soil_spatial'].items()[0][1])
        except (TypeError, KeyError, IndexError):
            zones = 1
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_landuse_zone_names(data, name):
    """Finalize the "macropore_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        try:
            zones = len(params['lu_spatial'].items()[0][1])
        except (TypeError, KeyError, IndexError):
            zones = 1
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))


###############################################################################
def fin_free_throughfall(data, name):
    """Finalize the "free_throughfall" parameter.

    1) if not provided, set it to all 1s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        default = 1.0
        data['params'][name] = dict((k, default) for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)


###############################################################################
def fin_max_canopy_storage(data, name):
    """Finalize the "max_canopy_storage" parameter.

    1) if not provided, set it to all 1s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        default = 0.0
        data['params'][name] = dict((k, default) for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)


###############################################################################
def fin_rapid_runoff_params(data, name):
    """Finalize the "max_canopy_storage" parameter.

    1) if not provided, set it to 0.
    """
    if data['params'][name] is None:
        data['params'][name] = [{'class_smd': [0],
                                 'class_ri': [0],
                                 'values': [[0.0], [0.0]]}]
    else:
        for dataset in data['params'][name]:
            dataset['values'] = [[float(i) for i in row] for row in
                                 dataset['values']]


###############################################################################
def fin_rorecharge_proportion(data, name):
    """Finalize the "rorecharge_proportion" parameter.

    1) if not provided, set it to 0.
    """
    params = data['params']
    if params[name] is None:
        params[name] = dict((k, [0.0]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params['ror_prop'] = sorted(params['rorecharge_proportion'].items(),
                                key=lambda x: x[0])
    params['ror_prop'] = np.array([i[1] for i in params['ror_prop']])


###############################################################################
def fin_rorecharge_limit(data, name):
    """Finalize the "rorecharge_limit" parameter.

    1) if not provided, set it to 99999.
    """
    params = data['params']
    if params[name] is None:
        params[name] = dict((k, [99999]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params['ror_limit'] = sorted(params['rorecharge_limit'].items(),
                                 key=lambda x: x[0])
    params['ror_limit'] = np.array([i[1] for i in params['ror_limit']])


###############################################################################
def fin_macropore_proportion(data, name):
    """Finalize the "macropore_proportion" parameter.

    1) if not provided, set it to 0.
    """
    params = data['params']
    if params[name] is None:
        params[name] = dict((k, [0.0]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params['macro_prop'] = sorted(params['macropore_proportion'].items(),
                                  key=lambda x: x[0])
    params['macro_prop'] = np.array([i[1] for i in params['macro_prop']])


###############################################################################
def fin_macropore_limit(data, name):
    """Finalize the "macropore_limit" parameter.

    1) if not provided, set it to 99999.
    """
    params = data['params']
    if params[name] is None:
        params[name] = dict((k, [99999.9]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999.9]', name)

    params['macro_limit'] = sorted(params['macropore_limit'].items(),
                                   key=lambda x: x[0])
    params['macro_limit'] = np.array([i[1] for i in params['macro_limit']])


###############################################################################
def fin_soil_static_params(data, name):
    """Finalize the "soil_static_params" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data['params']
    if params[name] is None and params['fao_process'] == 'enabled' and \
            params['fao_input'] == 'ls':
        params['fao_process'] = 'disabled'
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)


###############################################################################
def fin_soil_spatial(data, name):
    """Finalize the "soil_spatial" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data['params']
    if params[name] is None and params['fao_process'] == 'enabled' and \
            params['fao_input'] == 'ls':
        params['fao_process'] = 'disabled'
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)


###############################################################################
def fin_lu_spatial(data, name):
    """Finalize the "lu_spatial" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data['params']
    if params[name] is None and params['fao_process'] == 'enabled':
        params['fao_process'] = 'disabled'
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)


###############################################################################
def fin_zr(data, name):
    """Finalize the "zr" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data['params']
    if params[name] is None and params['fao_process'] == 'enabled' and \
            params['fao_input'] == 'ls':
        params['fao_process'] = 'disabled'
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)


###############################################################################
def fin_kc(data, name):
    """Finalize the "kc" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data['params']
    if params[name] is None and params['fao_process'] == 'enabled' and \
            params['fao_input'] == 'ls':
        params['fao_process'] = 'disabled'
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)
    elif params[name]:
        params['kc_list'] = sorted(params[name].items(), key=lambda x: x[0])
        params['kc_list'] = np.array([i[1] for i in params['kc_list']])


###############################################################################
def fin_taw_and_raw(data, name):
    """Finalize the "taw" and "raw" parameters."""
    params = data['params']
    if params['taw'] is None and params['fao_input'] == 'l':
        params['fao_input'] = 'ls'
        logging.info('\t\tSwitched "fao_input" to "ls", "taw" is missing')

    if params['raw'] is None and params['fao_input'] == 'l':
        params['fao_input'] = 'ls'
        logging.info('\t\tSwitched "fao_input" to "ls", "raw" is missing')

    if params['fao_input'] == 'ls':
        params['taw'], params['raw'] = u.build_taw_raw(params)
        logging.info('\t\tInferred "taw" and "raw" from soil params')

    elif params['fao_input'] == 'l':
        params['taw'] = u.invert_taw_raw(params['taw'], params)
        params['raw'] = u.invert_taw_raw(params['raw'], params)

    if params['taw'] is not None and params['raw'] is not None:
        for node in range(1, params['num_nodes'] + 1):
            params['taw'][node] = np.array(params['taw'][node]).astype(float)
            params['raw'][node] = np.array(params['raw'][node]).astype(float)


###############################################################################
def fin_subsoilzone_leakage_fraction(data, name):
    """Finalize the "subsoilzone_leakage_fraction" parameter.

    1) if not provided, set it to all 0s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        default = 0.0
        data['params'][name] = dict((k, default) for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)


###############################################################################
def fin_interflow_params(data, name):
    """Finalize the "interflow_params" parameter.

    1) if not provided, set it to all [0, 1, 999999, 0].
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, [0, 1, 999999, 0]) for k in
                                    range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, [0, 1, 999999, 0])


###############################################################################
def fin_recharge_attenuation_params(data, name):
    """Finalize the "recharge_attenuation_params" parameter.

    1) if not provided, set it to all [0, 1, 999999].
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, [0, 1, 999999]) for k in
                                    range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, [0, 1, 999999])


###############################################################################
def fin_date(data, name):
    """Finalize the "date" series."""
    series, params = data['series'], data['params']
    max_time = max([i for j in params['time_periods'] for i in j]) - 1
    day = datetime.timedelta(1)
    series['date'] = [params['start_date'] + day * num for num in
                      range(max_time)]
    dates = np.array([np.datetime64(str(i.date())) for i in series['date']])
    series['months'] = dates.astype('datetime64[M]').astype(int) % 12


###############################################################################
def fin_rainfall_ts(data, name):
    """Finalize the "rainfall_ts" series."""
    series = data['series']
    series[name] = np.array(series[name])


###############################################################################
def fin_pe_ts(data, name):
    """Finalize the "pe_ts" series."""
    series, specs, params = data['series'], data['specs'], data['params']

    fao = params['fao_process']
    canopy = params['canopy_process']
    if fao != 'enabled' and canopy != 'enabled':
        zones = len(set(params['pe_zone_mapping'].values()))
        series[name] = np.zeros([len(series['date']), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)
    elif not specs[name]['required']:
        specs[name]['required'] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)


###############################################################################
def fin_temperature_ts(data, name):
    """Finalize the "temperature_ts" series."""
    series, specs, params = data['series'], data['specs'], data['params']

    if params['snow_process'] == 'enabled' and not specs[name]['required']:
        specs[name]['required'] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        zones = len(set(params['temperature_zone_mapping'].values()))
        series[name] = np.zeros([len(series['date']), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)


###############################################################################
def fin_subroot_leakage_ts(data, name):
    """Finalize the "subroot_leakage_ts" series."""
    series, specs, params = data['series'], data['specs'], data['params']

    if params['leakage_process'] == 'enabled' and not specs[name]['required']:
        specs[name]['required'] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        values = [i[0] for i in params['subroot_zone_mapping'].values()]
        zones = len(set(values))
        series[name] = np.zeros([len(series['date']), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)


FUNC_PARAMS = [fin_start_date,
               fin_run_name,
               fin_num_cores,
               fin_output_recharge,
               fin_output_individual,
               fin_irchcb,
               fin_nodes_per_line,
               fin_reporting_zone_mapping,
               fin_reporting_zone_names,
               fin_rainfall_zone_names,
               fin_pe_zone_names,
               fin_temperature_zone_mapping,
               fin_temperature_zone_names,
               fin_subroot_zone_mapping,
               fin_subroot_zone_names,
               fin_rapid_runoff_zone_mapping,
               fin_rapid_runoff_zone_names,
               fin_rorecharge_zone_mapping,
               fin_rorecharge_zone_names,
               fin_macropore_zone_mapping,
               fin_macropore_zone_names,
               fin_soil_zone_names,
               fin_landuse_zone_names,
               fin_free_throughfall,
               fin_max_canopy_storage,
               fin_rapid_runoff_params,
               fin_rorecharge_proportion,
               fin_rorecharge_limit,
               fin_macropore_proportion,
               fin_macropore_limit,
               fin_soil_static_params,
               fin_soil_spatial,
               fin_lu_spatial,
               fin_taw_and_raw,
               fin_zr,
               fin_kc,
               fin_subsoilzone_leakage_fraction,
               fin_interflow_params,
               fin_recharge_attenuation_params]

FUNC_SERIES = [fin_date,
               fin_rainfall_ts,
               fin_pe_ts,
               fin_temperature_ts,
               fin_subroot_leakage_ts]


###############################################################################
def finalize_params(data):
    """Finalize all parameters."""
    logging.info('\tFinalizing parameters')

    for function in FUNC_PARAMS:
        param = function.__name__.replace('fin_', '')
        try:
            function(data, param)
        except Exception as err:
            raise u.FinalizationError('Could not finalize "%s": %s' %
                                      (param, err))
        logging.debug('\t\t"%s" finalized', param)

    logging.info('\tDone.')


###############################################################################
def finalize_series(data):
    """Finalize all time series."""
    logging.info('\tFinalizing time series')

    for function in FUNC_SERIES:
        series = function.__name__.replace('fin_', '')
        try:
            function(data, series)
        except Exception as err:
            raise u.FinalizationError('Could not finalize "%s": %s' %
                                      (series, err))
        logging.debug('\t\t"%s" finalized', series)

    logging.info('\tDone.')
