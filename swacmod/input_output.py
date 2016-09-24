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
import numpy as np

# Internal modules
from . import utils as u
from . import checks as c
from . import validation as v
from . import finalization as f


###############################################################################
def start_logging(level=logging.INFO, path=None):
    """Start logging output."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = ('%(asctime)s --- (%(process)d) %(levelname)s - %(message)s')

    if path is None:
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], '%s.log' % now)

    logging.basicConfig(filename=path,
                        format=log_format,
                        level=level)

    return path


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
            string = '%.6e' % (value/1000.0)
        else:
            string = '%.5e' % (value/1000.0)
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
                if len(row) < data['params']['nodes_per_line']:
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
                    return {}
                params[param] = dict((row[0], row[1]) for row in reader)
            elif ext == 'yml':
                try:
                    params[param] = load_yaml(absolute)[param]
                except (IOError, KeyError) as err:
                    print '---> Validation failed: %s (%s)' % (err, param)
                    return {}

    for key in specs:
        if key not in params:
            params[key] = None

    series = {}
    keys = [i for i in params if i.endswith('_ts')]
    for key in keys:
        series[key] = np.array(params.pop(key))

    data = {'specs': specs,
            'series': series,
            'params': params}

    f.finalize_params(data)
    f.finalize_series(data)

    return data


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
