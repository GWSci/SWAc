#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod input/output functions."""

# Standard Library
import os
import csv
import logging
import datetime

# Internal modules
from . import utils as u


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
def dump_output(data):
    """Write output to file."""
    path = os.path.join(u.CONSTANTS['OUTPUT_DIR'], 'output.csv')
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
def load_params_from_yaml():
    """Load model parameters."""
    pass


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

    params['rainfall_to_runoff'] = {
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
    columns = ['date', 'rainfall', 'PE', 'T', 'SZL']
    series = dict((k, []) for k in columns)

    for row in range(1, sheet.nrows):
        values = [i.value for i in sheet.row(row)]
        values[0] = u.convert_cell_to_date(values[0])
        series['date'].append(values[0])
        series['rainfall'].append(values[1])
        series['PE'].append(values[2])
        series['T'].append(values[3])
        series['SZL'].append(values[4])

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
