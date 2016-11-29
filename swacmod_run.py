#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import os
import sys
import time
import logging
import argparse
from multiprocessing import Process, Manager

# Third Party Libraries
import numpy as np

# Internal modules
from swacmod import utils as u
from swacmod import input_output as io

# Compile and import model
u.compile_model()
from swacmod import model as m


###############################################################################
def get_output(data, node):
    """Run the model."""
    logging.debug('\tRunning model for node %d', node)

    start = time.time()

    zone_r = data['params']['rainfall_zone_mapping'][node][0] - 1
    zone_p = data['params']['pe_zone_mapping'][node][0] - 1
    output = {'rainfall_ts': data['series']['rainfall_ts'][:, zone_r],
              'pe_ts': data['series']['pe_ts'][:, zone_p]}

    for function in [m.get_pefac,
                     m.get_canopy_storage,
                     m.get_net_pefac,
                     m.get_precip_to_ground,
                     m.get_snowfall_o,
                     m.get_rainfall_o,
                     m.get_snow,
                     m.get_net_rainfall,
                     m.get_rawrew,
                     m.get_tawrew,
                     m.get_ae,
                     m.get_unutilised_pe,
                     m.get_perc_through_root,
                     m.get_subroot_leak,
                     m.get_interflow_bypass,
                     m.get_interflow_store_input,
                     m.get_interflow,
                     m.get_recharge_store_input,
                     m.get_recharge,
                     m.get_combined_str,
                     m.get_combined_ae,
                     m.get_evt,
                     m.get_average_in,
                     m.get_average_out,
                     m.get_change,
                     m.get_balance]:

        columns = function(data, output, node)
        output.update(columns)
        logging.debug('\t\t"%s()" done', function.__name__)

    end = time.time()
    logging.debug('\tNode %d done (%dms).', node, (end - start) * 1000)
    return output


###############################################################################
def run_process(num, ids, data, test, reporting, recharge, log_path, level,
                file_format):
    """Run model for a chunk of nodes."""
    io.start_logging(path=log_path, level=level)
    logging.info('Process %d started (%d nodes)', num, len(ids))
    for node in ids:
        rep_zone = data['params']['reporting_zone_mapping'][node]
        if rep_zone == 0:
            continue
        output = get_output(data, node)
        logging.debug('RAM usage is %.2fMb', u.get_ram_usage_for_process())
        if not test:
            if node in data['params']['output_individual']:
                io.dump_water_balance(data, output, file_format, node=node)
            if rep_zone not in reporting:
                reporting[rep_zone] = output.copy()
            else:
                area = data['params']['node_areas'][node]
                reporting[rep_zone] = m.aggregate(reporting[rep_zone], output,
                                                  area)
            recharge[node] = output['combined_recharge'].copy()
    logging.info('Process %d ended', num)


###############################################################################
def run(test=False, debug=False, file_format=None):
    """Run model for all nodes."""
    print '\nInput: %s' % u.CONSTANTS['INPUT_DIR']
    print 'Output: %s\n' % u.CONSTANTS['OUTPUT_DIR']

    level = (logging.DEBUG if debug else logging.INFO)
    log_path = io.start_logging(level=level)
    logging.info('Start SWAcMod run')

    manager = Manager()
    reporting = manager.dict()
    recharge = manager.dict()

    specs_file = u.CONSTANTS['SPECS_FILE']
    if test:
        input_file = u.CONSTANTS['TEST_INPUT_FILE']
        input_dir = u.CONSTANTS['TEST_INPUT_DIR']
    else:
        input_file = u.CONSTANTS['INPUT_FILE']
        input_dir = u.CONSTANTS['INPUT_DIR']

    data = io.load_and_validate(specs_file, input_file, input_dir)

    ids = range(1, data['params']['num_nodes'] + 1)
    chunks = np.array_split(ids, data['params']['num_cores'])

    procs = {}
    for num, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        procs[num] = Process(target=run_process,
                             args=(num, chunk, data, test, reporting,
                                   recharge, log_path, level, file_format))
        procs[num].start()

    for num in procs:
        procs[num].join()

    if not test:
        for key in reporting.keys():
            io.dump_water_balance(data, reporting[key], file_format, zone=key)
        if data['params']['output_recharge']:
            io.dump_recharge_file(data, recharge)

    logging.info('End SWAcMod run')


###############################################################################
if __name__ == "__main__":

    # Parser for command line arguments
    DESCRIPTION = """
    Invoke this script to run SWAcMod.
    e.g. 'python swacmod_run.py'"""
    FORM = argparse.RawTextHelpFormatter

    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    PARSER.add_argument('-t',
                        '--test',
                        help='run with no output',
                        action='store_true')
    PARSER.add_argument('-d',
                        '--debug',
                        help='verbose log',
                        action='store_true')
    PARSER.add_argument('-i',
                        '--input_dir',
                        help='path to input directory')
    PARSER.add_argument('-o',
                        '--output_dir',
                        help='path to output directory')
    PARSER.add_argument('-f',
                        '--format',
                        help='output file format',
                        choices=['hdf5', 'csv'],
                        default='csv')

    ARGS = PARSER.parse_args()
    if ARGS.input_dir:
        u.CONSTANTS['INPUT_DIR'] = ARGS.input_dir
        u.CONSTANTS['INPUT_FILE'] = os.path.join(ARGS.input_dir, 'input.yml')
    if ARGS.output_dir:
        u.CONSTANTS['OUTPUT_DIR'] = ARGS.output_dir

    run(test=ARGS.test, debug=ARGS.debug, file_format=ARGS.format)

