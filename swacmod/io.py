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
    logging.info('\t\tLoading %s', filein)

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
def dump_output(data, node):
    """Write output to file."""
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], 'output_%d.csv' % node)
    logging.info('\tDumping output to "%s"', path)

    with open(path, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(u.CONSTANTS['COL_ORDER'])
        for num in range(len(data['series']['date'])):
            row = []
            for key in u.CONSTANTS['COL_ORDER']:
                try:
                    row.append(data['output'][key][num])
                except KeyError:
                    continue
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
    max_time = max([i for j in params['time_periods'].values() for i in j])
    day = datetime.timedelta(1)
    series['date'] = [params['start_date'] + day * num for num in
                      range(max_time)]


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


###############################################################################
def finalize_pe_ts(specs, params, series):
    """Finalize the "pe_ts" series i/o."""
    fao = params['fao_process']
    canopy = params['canopy_process']
    if fao != 'enabled' and canopy != 'enabled':
        series['pe_ts'] = [0.0 for _ in series['date']]
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
        series[key] = params.pop(key)

    logging.info('\tFinalize load')
    finalize_start_date(params)
    finalize_date(params, series)
    finalize_taw_and_raw(params)
    finalize_pe_ts(specs, params, series)
    finalize_temperature_ts(specs, params)
    finalize_subroot_leakage_ts(specs, params)

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
