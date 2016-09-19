#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod input/output functions."""

# Standard Library
import os
import re
import csv
import sys
import logging
import datetime

# Third Party Libraries
import yaml
from dateutil import parser
import numpy as np

# Internal modules
from . import utils as u
from . import checks as c
from . import validation as v


###############################################################################
def start_logging(level=logging.INFO):
    """Start logging output."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], '%s.log' % now)
    log_format = ('%(asctime)s --- (%(process)d) %(levelname)s - %(message)s')

    logging.basicConfig(filename=path,
                        format=log_format,
                        level=level)


###############################################################################
def load_yaml(filein):
    """Load a YAML file, lowercase its keys."""
    logging.debug('\t\tLoading %s', filein)

    yml = yaml.load(open(filein, 'r'), Loader=yaml.CLoader)
    try:
        keys = yml.keys()
    except AttributeError:
        return yml

    for key in keys:
        if not key.islower():
            new_key = key.lower()
            value = yml.pop(key)
            yml[new_key] = value

    return yml


###############################################################################
def format_recharge_row(row):
    """Convert list of values to output string."""
    final = []
    for value in row:
        if value >= 0:
            string = '%.6e' % value/1000.0
        else:
            string = '%.5e' % value/1000.0
        splitter = ('e+' if 'e+' in string else 'e-')
        split = string.split(splitter)
        if len(split[1]) == 2:
            split[1] = '0%s' % split[1]
        final.append(splitter.join(split))
    return ' '.join(final) + '\n'


###############################################################################
def dump_recharge_file(data, recharge):
    """Write recharge to file."""
    nrchop, inrech = 3, 1

    fileout = '%s_recharge.rch' % data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    logging.debug('\tDumping recharge to "%s"', path)

    with open(path, 'w') as rech_file:
        rech_file.write('# MODFLOW-USGs Recharge Package\n')
        rech_file.write(' %d %d\n' % (nrchop, data['params']['irchcb']))

        final = {}
        for node in recharge.keys():
            final[node] = u.aggregate_output_col(data,
                                                 {'recharge': recharge[node]},
                                                 'recharge', method='average')

        for num in range(len(data['params']['time_periods'])):
            rech_file.write(' %d\n' % inrech)
            rech_file.write('INTERNAL  1.000000e+000  (FREE)  -1  RECHARGE\n')
            row = []
            for node in sorted(recharge.keys()):
                if len(row) < data['params']['irchcb']:
                    row.append(final[node][num])
                else:
                    rech_file.write(format_recharge_row(row))
                    row = []
            if row:
                rech_file.write(format_recharge_row(row))


###############################################################################
def dump_water_balance(data, output, node=None, zone=None):
    """Write output to file."""
    areas = data['params']['node_areas']
    periods = data['params']['time_periods']

    if node:
        fileout = 'output_node_%d.csv' % node
        area = areas[node]
    elif zone:
        fileout = 'output_zone_%d.csv' % zone
        items = data['params']['reporting_zone_mapping'].items()
        area = sum([areas[i[0]] for i in items if i[1] == zone])

    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    logging.debug('\tDumping output to "%s"', path)
    aggregated = u.aggregate_output(data, output, method='sum')

    with open(path, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([i[0] for i in u.CONSTANTS['BALANCE_CONVERSIONS']])
        for num in range(len(periods)):
            row = [aggregated[key][num] for key in u.CONSTANTS['COL_ORDER'] if
                   key not in ['unutilised_pe', 'k_slope', 'rapid_runoff_c']]
            row.insert(1, periods[num][1] - periods[num][0] + 1)
            row.insert(2, area)
            for num2, element in enumerate(row):
                if u.CONSTANTS['BALANCE_CONVERSIONS'][num][1]:
                    row[num2] = element / 1000.0 * area
            writer.writerow(row)


###############################################################################
def convert_all_yaml_to_csv(specs_file=u.CONSTANTS['SPECS_FILE'],
                            input_dir=u.CONSTANTS['INPUT_DIR']):
    """Convert all YAML files to CSV for parameters that accept this option."""
    specs = load_yaml(specs_file)
    to_csv = [i for i in specs if 'alt_format' in specs[i] and 'csv' in
              specs[i]['alt_format']]
    for filein in os.listdir(input_dir):
        if filein.endswith('.yml'):
            path = os.path.join(input_dir, filein)
            loaded = load_yaml(path)
            param = loaded.items()[0][0]
            if len(loaded.items()) == 1 and param in to_csv:
                print path
                convert_one_yaml_to_csv(path)


###############################################################################
def convert_one_yaml_to_csv(filein):
    """Convert a YAML file to a CSV file.

    The opposite function is tricky, as CSV reader does not understand types.
    """
    fileout = filein.replace('.yml', '.csv')
    readin = load_yaml(filein).items()[0][1]

    with open(fileout, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if isinstance(readin, dict):
            for item in readin.items():
                row = [item[0]]
                if isinstance(item[1], list):
                    row += item[1]
                elif isinstance(item[1], (float, int, long, str)):
                    row += [item[1]]
                else:
                    print 'Could not recognize object: %s' % type(item[1])
                writer.writerow(row)
        elif isinstance(readin, list):
            for item in readin:
                row = []
                if isinstance(item, list):
                    row += item
                elif isinstance(item, (float, int, long, str)):
                    row += [item]
                else:
                    print 'Could not recognize object: %s' % type(item)
                writer.writerow(row)


###############################################################################
def finalize_start_date(params):
    """Finalize the "start_date" parameter i/o."""
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
def finalize_date(params, series):
    """Finalize the "date" series i/o."""
    max_time = max([i for j in params['time_periods'] for i in j])
    day = datetime.timedelta(1)
    series['date'] = [params['start_date'] + day * num for num in
                      range(max_time)]
    dates = np.array([np.datetime64(str(i.date())) for i in series['date']])
    series['months'] = dates.astype('datetime64[M]').astype(int) % 12


###############################################################################
def finalize_output_individual(params):
    """Finalize the "output_individual" parameter i/o."""
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
def finalize_taw_and_raw(params):
    """Finalize the "TAW" and "RAW" parameters i/o."""
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
def finalize_pe_ts(specs, params, series):
    """Finalize the "pe_ts" series i/o."""
    fao = params['fao_process']
    canopy = params['canopy_process']
    if fao != 'enabled' and canopy != 'enabled':
        series['pe_ts'] = np.zeros(len(series['date']))
        logging.info('\t\tDefaulted "pe_ts" to 0.0')
    else:
        specs['pe_ts']['required'] = True
        logging.info('\t\tSwitched "pe_ts" to "required"')


###############################################################################
def finalize_temperature_ts(specs, params):
    """Finalize the "temperature_ts" series i/o."""
    if params['snow_process'] == 'enabled':
        specs['temperature_ts']['required'] = True
        logging.info('\t\tSwitched "temperature_ts" to "required"')


###############################################################################
def finalize_subroot_leakage_ts(specs, params):
    """Finalize the "subroot_leakage_ts" series i/o."""
    if params['leakage_process'] == 'enabled':
        specs['subroot_leakage_ts']['required'] = True
        logging.info('\t\tSwitched "subroot_leakage_ts" to "required"')


###############################################################################
def finalize_numpy_arrays(params):
    """Convert dictionaries to numpy arrays for efficiency."""
    params['kc_list'] = sorted(params['kc'].items(), key=lambda x: x[0])
    params['kc_list'] = np.array([i[1] for i in params['kc_list']])

    params['ror_prop'] = sorted(params['rorecharge_proportion'].items(),
                                key=lambda x: x[0])
    params['ror_prop'] = np.array([i[1] for i in params['ror_prop']])

    params['ror_limit'] = sorted(params['rorecharge_limit'].items(),
                                 key=lambda x: x[0])
    params['ror_limit'] = np.array([i[1] for i in params['ror_limit']])

    params['macro_prop'] = sorted(params['macropore_proportion'].items(),
                                  key=lambda x: x[0])
    params['macro_prop'] = np.array([i[1] for i in params['macro_prop']])

    params['macro_limit'] = sorted(params['macropore_limit'].items(),
                                   key=lambda x: x[0])
    params['macro_limit'] = np.array([i[1] for i in params['macro_limit']])


###############################################################################
def load_params_from_yaml(specs_file=u.CONSTANTS['SPECS_FILE'],
                          input_file=u.CONSTANTS['INPUT_FILE'],
                          input_dir=u.CONSTANTS['INPUT_DIR']):
    """Load model specifications, parameters and time series."""
    logging.info('\tLoading parameters and time series')

    specs = load_yaml(specs_file)
    params = load_yaml(input_file)

    for param in params:
        if isinstance(params[param], str) and 'alt_format' in specs[param]:
            absolute = os.path.join(input_dir, params[param])
            ext = params[param].split('.')[-1]
            if ext not in specs[param]['alt_format']:
                continue
            if ext == 'csv':
                try:
                    reader = csv.reader(open(absolute, 'r'))
                except IOError as err:
                    print '---> Validation failed: %s (%s)' % (err, param)
                    return None, None, None
                params[param] = dict((row[0], row[1]) for row in reader)
            elif ext == 'yml':
                try:
                    params[param] = load_yaml(absolute)[param]
                except (IOError, KeyError) as err:
                    print '---> Validation failed: %s (%s)' % (err, param)
                    return None, None, None

    for key in specs:
        if key not in params:
            params[key] = None

    series = {}
    keys = [i for i in params if i.endswith('_ts')]
    for key in keys:
        series[key] = np.array(params.pop(key))

    logging.info('\tFinalize load')
    finalize_start_date(params)
    finalize_date(params, series)
    finalize_output_individual(params)
    finalize_taw_and_raw(params)
    finalize_pe_ts(specs, params, series)
    finalize_temperature_ts(specs, params)
    finalize_subroot_leakage_ts(specs, params)
    finalize_numpy_arrays(params)

    return specs, series, params


###############################################################################
def validate_all(data):
    """Validate model parameters and time series."""
    try:
        c.check_required(data)
        v.validate_params(data)
        v.validate_series(data)
    except u.ValidationError as err:
        print '---> Validation failed: %s' % err
        sys.exit()


###############################################################################
def load_results():
    """Load 'Calculations' sheet."""
    check = dict((k, []) for k in u.CONSTANTS['COL_ORDER'])

    with open(u.CONSTANTS['TEST_RESULTS_FILE'], 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            values = []
            for num, cell in enumerate(row):
                if num == 0:
                    values.append(parser.parse(cell))
                else:
                    values.append(float(cell))
            for num, value in enumerate(values):
                check[u.CONSTANTS['COL_ORDER'][num]].append(value)

    return check
