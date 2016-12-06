#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod input/output functions."""

# Standard Library
import os
import csv
import ast
import sys
import logging
import datetime

# Third Party Libraries
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import pandas as pd
from dateutil import parser

# Internal modules
from . import utils as u
from . import checks as c
from . import validation as v
from . import finalization as f


###############################################################################
def start_logging(level=logging.INFO, path=None, run_name=None):
    """Start logging output.

    If path is None, run_name has to be provided.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = ('%(asctime)s --- (%(process)d) %(levelname)s - %(message)s')

    disc = '''
    Although this program has been subjected to rigorous review, Groundwater
    Science Ltd. reserves the right to update the software as needed pursuant
    to further analysis and review. No warranty, expressed or implied, is made
    by Groundwater Science Ltd. as to the functionality of the software nor
    shall the fact of release constitute any such warranty. Furthermore, the
    tool is released on condition that Groundwater Science Ltd. shall not be
    held liable for any damages resulting from its authorised or unauthorised
    use.
    '''

    if path is None:
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        name = '%s_%s.log' % (run_name, now)
        path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], name)
        logging.basicConfig(filename=path,
                            format='%(message)s',
                            level=level)
        logging.info(disc)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=path,
                        format=log_format,
                        level=level)

    return path


###############################################################################
def format_time(diff):
    """Format a time difference for output."""
    secs = int(round(diff))
    return str(datetime.timedelta(0, secs))


###############################################################################
def print_progress(nodes, total):
    """Start logging output.

    If path is None, run_name has to be provided.
    """
    perc = nodes * 1.0 / total
    perc_big = perc * 100
    spaces = int(perc_big / 2)
    progress = '=' * spaces + '>' + ' ' * (50 - spaces)

    sys.stdout.write('Run SWAcMod: [%s] %d%%\r' % (progress, perc_big))
    sys.stdout.flush()

    if nodes == total:
        print


###############################################################################
def load_yaml(filein):
    """Load a YAML file, lowercase its keys."""
    logging.debug('\t\tLoading %s', filein)

    yml = yaml.load(open(filein, 'r'), Loader=Loader)
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
def get_recharge_path(data):
    """Return the path of the recharge file."""
    fileout = '%s_recharge.rch' % data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    return path


###############################################################################
def get_output_path(data, file_format, node=None, zone=None):
    """Return the path of the recharge file."""
    run = data['params']['run_name']

    if node:
        _ = len(str(data['params']['num_nodes']))
        counter = eval("'%%0%dd' % _") % node
        fileout = '%s_n_%s.%s' % (run, counter, file_format)
    elif zone:
        zones = data['params']['reporting_zone_mapping'].values()
        _ = len(str(len(set(zones))))
        counter = eval("'%%0%dd' % _") % zone
        fileout = '%s_z_%s.%s' % (run, counter, file_format)

    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    return path


###############################################################################
def check_open_files(data, file_format):
    """Check if any of the scheduled output files can't be open."""
    paths = []
    if data['params']['output_recharge']:
        paths.append(get_recharge_path(data))

    for node in data['params']['output_individual']:
        paths.append(get_output_path(data, file_format, node=node))

    zones = set(data['params']['reporting_zone_mapping'].values())
    for zone in zones:
        paths.append(get_output_path(data, file_format, zone=zone))

    for path in paths:
        try:
            fileobj = open(path, 'w')
            fileobj.close()
        except IOError:
            print ('\nCannot write to "%s", make sure the file is not '
                   'in use\n' % path)
            sys.exit()


###############################################################################
def dump_recharge_file(data, recharge):
    """Write recharge to file."""
    nrchop, inrech = 3, 1

    fileout = '%s_recharge.rch' % data['params']['run_name']
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], fileout)
    logging.info('\tDumping recharge file')

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
def dump_water_balance(data, output, file_format, node=None, zone=None,
                       reduced=False):
    """Write output to file."""
    areas = data['params']['node_areas']
    periods = data['params']['time_periods']

    if node:
        string = 'for node %d' % node
        area = areas[node]
    elif zone:
        string = 'for zone %d' % zone
        items = data['params']['reporting_zone_mapping'].items()
        area = sum([areas[i[0]] for i in items if i[1] == zone])

    path = get_output_path(data, file_format, node=node, zone=zone)
    logging.info('\tDumping water balance %s', string)
    aggregated = u.aggregate_output(data, output, method='sum')

    with open(path, 'wb') as outfile:

        if reduced:
            header = [i[0] for i in u.CONSTANTS['BALANCE_CONVERSIONS'] if i[2]]
        else:
            header = [i[0] for i in u.CONSTANTS['BALANCE_CONVERSIONS']]
        if file_format == 'csv':
            writer = csv.writer(outfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
        elif file_format == 'hdf5':
            writer = pd.HDFStore(path)
        final = []
        for num, period in enumerate(periods):
            if reduced:
                row = [aggregated[key][num] for key in
                       ['date', 'unutilised_pe', 'combined_recharge',
                        'combined_str', 'combined_ae']]
            else:
                row = [aggregated[key][num] for key in u.CONSTANTS['COL_ORDER']
                       if key not in ['unutilised_pe', 'k_slope',
                                      'rapid_runoff_c']]
            row.insert(1, period[1] - period[0])
            row.insert(2, area)
            for num2, element in enumerate(row):
                if u.CONSTANTS['BALANCE_CONVERSIONS'][num2][1]:
                    if node:
                        row[num2] = element / 1000.0 * area
                    elif zone:
                        row[num2] = element / 1000.0
            if file_format == 'csv':
                writer.writerow(row)
            elif file_format == 'hdf5':
                final.append(row)

        if file_format == 'hdf5':
            data_frame = pd.DataFrame(final, columns=header)
            writer.append('data', data_frame, index=False)
            writer.close()


###############################################################################
def convert_all_yaml_to_csv(specs_file, input_dir):
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

    no_list = ['node_areas', 'free_throughfall', 'max_canopy_storage',
               'subsoilzone_leakage_fraction'] + \
              [i for i in params if 'zone_names' in i] + \
              [i for i in params if 'zone_mapping' in i and i not in
               ['rainfall_zone_mapping', 'pe_zone_mapping',
                'subroot_zone_mapping']]

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
                    msg = 'Could not import %s: %s' % (param, err)
                    raise u.InputOutputError(msg)
                try:
                    rows = [[ast.literal_eval(j) for j in row]
                            for row in reader]
                    if param.endswith('_ts') or param == 'time_periods':
                        params[param] = rows
                    else:
                        if param not in no_list:
                            params[param] = dict((row[0], row[1:]) for row in
                                                 rows)
                        else:
                            params[param] = dict((row[0], row[1]) for row in
                                                 rows)
                except IndexError as err:
                    msg = 'Could not import %s: %s' % (param, err)
                    raise u.InputOutputError(msg)
            elif ext == 'yml':
                try:
                    params[param] = load_yaml(absolute)[param]
                except (IOError, KeyError) as err:
                    msg = 'Could not import %s: %s' % (param, err)
                    raise u.InputOutputError(msg)

    for key in specs:
        if key not in params:
            params[key] = None

    series = {}
    keys = [i for i in params if i.endswith('_ts')]
    for key in keys:
        series[key] = params.pop(key)

    data = {'specs': specs,
            'series': series,
            'params': params}

    return data


###############################################################################
def load_and_validate(specs_file, input_file, input_dir):
    """Load, finalize and validate model parameters and time series."""
    try:
        data = load_params_from_yaml(specs_file=specs_file,
                                     input_file=input_file,
                                     input_dir=input_dir)
    except u.InputOutputError as err:
        print '---> InputOutput failed: %s' % err
        sys.exit()

    try:
        f.finalize_params(data)
        f.finalize_series(data)
    except u.FinalizationError as err:
        print '---> Finalization failed: %s' % err
        sys.exit()

    try:
        c.check_required(data)
        v.validate_params(data)
        v.validate_series(data)
    except u.ValidationError as err:
        print '---> Validation failed: %s' % err
        sys.exit()

    return data


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
