#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod utils."""

# Standard Library
import os
import datetime

# Third Party Libraries
import xlrd


CONSTANTS = {}

CONSTANTS['CODE_DIR'] = os.path.dirname(os.path.abspath(__file__))
CONSTANTS['ROOT_DIR'] = os.path.join(CONSTANTS['CODE_DIR'], '../')
CONSTANTS['INPUT_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'input_files')
CONSTANTS['OUTPUT_DIR'] = os.path.join(CONSTANTS['ROOT_DIR'], 'output_files')
CONSTANTS['EXCEL_PATH'] = os.path.join(CONSTANTS['INPUT_DIR'], 'input.xls')
CONSTANTS['EXCEL_BOOK'] = xlrd.open_workbook(CONSTANTS['EXCEL_PATH'])
CONSTANTS['INI_FILE'] = os.path.join(CONSTANTS['INPUT_DIR'], 'input.yml')

CONSTANTS['COL_ORDER'] = [
    'date', '', 'rainfall', 'PE', 'pefac', 'canopy_storage',
    'veg_diff', 'precipitation', 'snowfall_o', 'rainfall_o', 'snowpack',
    'snowmelt', 'net_rainfall', 'rapid_runoff_c', 'rapid_runoff',
    'runoff_recharge', 'macropore', 'perc_in_root', 'rawrew',
    'tawrew', 'p_smd', 'smd', 'k_s', 'ae', 'unutilized_pe',
    'perc_through_root', 'subroot_leak', 'interflow_bypass',
    'interflow_input', 'interflow_volume', 'infiltration_recharge',
    'interflow_to_rivers', 'recharge_input', 'recharge_store',
    'combined_recharge', 'str', 'combined_ae', 'evt', 'average_in',
    'average_out', 'balance'
]


###############################################################################
def convert_cell_to_date(value):
    """Convert a cell in an Excel Spreadsheet to a Python datetime object."""
    date = xlrd.xldate_as_tuple(value, CONSTANTS['EXCEL_BOOK'].datemode)
    return datetime.datetime(date[0], date[1], date[2])
