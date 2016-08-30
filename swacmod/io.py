#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod input/output functions."""

# Standard Library
import os
import csv
import sys
import logging
import datetime

# Third Party Libraries
import yaml
from dateutil import parser

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
def dump_output(data, node):
    """Write output to file."""
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], 'output_%d.csv' % node)
    logging.info('\tDumping output to "%s"', path)

    with open(path, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        order = [i for i in u.CONSTANTS['COL_ORDER'] if i != '']
        writer.writerow(order)
        for num in range(len(data['series']['rainfall'])):
            row = []
            for key in order:
                try:
                    row.append(data['series'][key][num])
                except KeyError:
                    row.append(data['output'][key][num])
            writer.writerow(row)


###############################################################################
def convert_all_yaml_to_csv():
    """Convert all YAML files to CSV for parameters that accept this option."""
    specs = yaml.load(open(u.CONSTANTS['SPECS_FILE'], 'r'))
    to_csv = [i for i in specs if 'alt_format' in specs[i] and 'csv' in
              specs[i]['alt_format']]
    for filein in os.listdir(u.CONSTANTS['INPUT_DIR']):
        if filein.endswith('.yml'):
            path = os.path.join(u.CONSTANTS['INPUT_DIR'], filein)
            loaded = yaml.load(open(path, 'r'))
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
    readin = yaml.load(open(filein, 'r')).items()[0][1]

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
def load_params_from_yaml(specs_file=u.CONSTANTS['SPECS_FILE'],
                          input_file=u.CONSTANTS['INPUT_FILE'],
                          input_dir=u.CONSTANTS['INPUT_DIR']):
    """Load model specifications, parameters and time series."""
    specs = yaml.load(open(specs_file, 'r'))
    params = yaml.load(open(input_file, 'r'))

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
                    print '---> Validation failed: %s' % err
                params[param] = dict((row[0], row[1]) for row in reader)
            elif ext == 'yml':
                try:
                    params[param] = yaml.load(open(absolute, 'r'))[param]
                except (IOError, KeyError) as err:
                    print '---> Validation failed: %s' % err

    series = {}
    keys = [i for i in params if i.endswith('_ts')]
    for key in keys:
        series[key] = params.pop(key)

    try:
        params['start_date'] = date = parser.parse(params['start_date'])
    except Exception as err:
        print '---> Validation failed: %s' % err

    max_time = max([i for j in params['time_periods'].values() for i in j])
    day = datetime.timedelta(1)
    series['date'] = [date + day * num for num in range(max_time)]

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
def load_params_from_excel():
    """Load model parameters."""
    logging.info('\tLoading parameters')
    sheet = u.CONSTANTS['EXCEL_BOOK'].sheets()[1]
    params = {}

    params['free_throughfall'] = sheet.row(2)[1].value
    params['max_canopy_storage'] = sheet.row(3)[1].value

    params['starting_snow_pack'] = sheet.row(6)[1].value
    params['snowfall_degrees'] = sheet.row(7)[1].value
    params['snowmelt_degrees'] = sheet.row(8)[1].value

    params['rapid_runoff_params'] = {
        'class_smd': [i.value for i in sheet.row(14)[2:7]],
        'class_ri': [sheet.row(i)[1].value for i in range(15, 22)],
        'values': [[i.value for i in sheet.row(j)[2:7]] for j in range(15, 22)]
    }

    col = [sheet.row(i)[2].value for i in range(26, 38)]
    params['recharge_proportion'] = col
    col = [sheet.row(i)[3].value for i in range(26, 38)]
    params['recharge_limit'] = col

    col = [sheet.row(i)[2].value for i in range(41, 53)]
    params['macropore_proportion'] = col
    col = [sheet.row(i)[3].value for i in range(41, 53)]
    params['macropore_limit'] = col

    params['FC'] = sheet.row(57)[1].value
    params['WP'] = sheet.row(58)[1].value
    params['p'] = sheet.row(59)[1].value

    params['KC_ini'] = {'month': sheet.row(57)[4].value,
                        'KC':    sheet.row(57)[5].value}
    params['KC_mid'] = {'month': sheet.row(58)[4].value,
                        'KC':    sheet.row(58)[5].value}
    params['KC_end'] = {'month': sheet.row(59)[4].value,
                        'KC':    sheet.row(59)[5].value}

    params['starting_SMD'] = sheet.row(56)[7].value

    col = [sheet.row(i)[2].value for i in range(63, 75)]
    params['ZR'] = col
    col = [sheet.row(i)[3].value for i in range(63, 75)]
    params['KC'] = col

    params['TAW'], params['RAW'] = [], []
    for num in range(len(params['ZR'])):
        var1 = 1000 * (params['FC'] - params['WP'])
        params['TAW'].append(var1 * params['ZR'][num])
        params['RAW'].append(params['TAW'][num] * params['p'])

    params['leakage'] = sheet.row(77)[1].value
    params['init_interflow_store'] = sheet.row(80)[3].value
    params['store_bypass'] = sheet.row(81)[3].value
    params['infiltration'] = sheet.row(82)[3].value
    params['interflow_to_rivers'] = sheet.row(83)[3].value

    params['init_recharge_store'] = sheet.row(87)[3].value
    params['release_proportion'] = sheet.row(88)[3].value
    params['release_limit'] = sheet.row(89)[3].value

    return params


###############################################################################
def load_input_from_excel():
    """Load input time series."""
    logging.info('\tLoading input time series')
    sheet = u.CONSTANTS['EXCEL_BOOK'].sheets()[0]
    columns = ['date', 'rainfall_ts', 'pe_ts', 'temperature_ts',
               'subroot_leakage_ts']
    series = dict((k, []) for k in columns)

    for row in range(1, sheet.nrows):
        values = [i.value for i in sheet.row(row)]
        values[0] = u.convert_cell_to_date(values[0])
        series['date'].append(values[0])
        series['rainfall_ts'].append(values[1])
        series['pe_ts'].append(values[2])
        series['temperature_ts'].append(values[3])
        series['subroot_leakage_ts'].append(values[4])

    return series


###############################################################################
def load_results():
    """Load 'Calculations' sheet."""
    sheet = u.CONSTANTS['EXCEL_BOOK'].sheets()[2]
    check = dict((k, []) for k in u.CONSTANTS['COL_ORDER'])

    for row in range(5, sheet.nrows):
        values = [i.value for i in sheet.row(row)][:-1]
        values[0] = u.convert_cell_to_date(values[0])
        for num, value in enumerate(values):
            check[u.CONSTANTS['COL_ORDER'][num]].append(value)

    return check
