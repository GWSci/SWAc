#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import os
import sys
import time
import logging
import argparse
from multiprocessing import Process, Manager, freeze_support

# Third Party Libraries
import numpy as np

# Internal modules
from swacmod import utils as u
from swacmod import input_output as io

# Compile and import model
u.compile_model()
from swacmod import model as m


###############################################################################
def aggregate_reporting(reporting):
    """Aggregate zones across processes."""
    logging.info('\tAggregating reporting across processes')
    new_rep = {}
    for key in reporting.keys():
        if key[1] not in new_rep:
            new_rep[key[1]] = reporting[key].copy()
        else:
            for key2 in reporting[key]:
                new_rep[key[1]][key2] += reporting[key][key2]
    return new_rep


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
                file_format, reduced, output_dir):
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
                io.dump_water_balance(data, output, file_format, output_dir,
                                      node=node, reduced=reduced)
            key = (num, rep_zone)
            area = data['params']['node_areas'][node]
            if key not in reporting:
                reporting[key] = m.aggregate(output, area)
            else:
                reporting[key] = m.aggregate(output, area,
                                             reporting=reporting[key])
            if data['params']['output_recharge']:
                recharge[node] = output['combined_recharge'].copy()
            else:
                recharge[node] = {}
            io.print_progress(len(recharge), data['params']['num_nodes'])
    logging.info('Process %d ended', num)


###############################################################################
def run(test=False, debug=False, file_format=None, reduced=False, skip=False):
    """Run model for all nodes."""
    times = {'start_of_run': time.time()}

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

    level = (logging.DEBUG if debug else logging.INFO)
    params = io.load_yaml(input_file)
    log_path = io.start_logging(level=level, run_name=params['run_name'])

    print '\nStart "%s"' % params['run_name']
    logging.info('Start SWAcMod run')
    print 'Loading parameters'

    data = io.load_and_validate(specs_file, input_file, input_dir)
    if not skip:
        io.check_open_files(data, file_format, u.CONSTANTS['OUTPUT_DIR'])

    ids = range(1, data['params']['num_nodes'] + 1)
    chunks = np.array_split(ids, data['params']['num_cores'])

    times['end_of_input'] = time.time()

    procs = {}
    for num, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        procs[num] = Process(target=run_process,
                             args=(num, chunk, data, test, reporting,
                                   recharge, log_path, level, file_format,
                                   reduced, u.CONSTANTS['OUTPUT_DIR']))
        procs[num].start()

    for num in procs:
        procs[num].join()

    times['end_of_model'] = time.time()

    if not test:
        print 'Writing output files'
        if not skip:
            io.check_open_files(data, file_format, u.CONSTANTS['OUTPUT_DIR'])
        reporting = aggregate_reporting(reporting)
        for key in reporting.keys():
            io.dump_water_balance(data, reporting[key], file_format,
                                  u.CONSTANTS['OUTPUT_DIR'], zone=key,
                                  reduced=reduced)
        if data['params']['output_recharge']:
            io.dump_recharge_file(data, recharge)

    times['end_of_run'] = time.time()

    diff = times['end_of_run'] - times['start_of_run']
    total = io.format_time(diff)
    per_node = int(round(diff * 1000/data['params']['num_nodes']))

    cores = ('%d cores' % data['params']['num_cores'] if
             data['params']['num_cores'] != 1 else '1 core')

    print '\nPerformance (%s)' % cores
    print 'Input time:  %s' % io.format_time(times['end_of_input'] -
                                             times['start_of_run'])
    print 'Run time:    %s' % io.format_time(times['end_of_model'] -
                                             times['end_of_input'])
    print 'Output time: %s' % io.format_time(times['end_of_run'] -
                                             times['end_of_model'])
    print 'Total time:  %s (%d msec/node)' % (total, per_node)
    print

    logging.info('End SWAcMod run')


###############################################################################
if __name__ == "__main__":
    freeze_support()

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
    PARSER.add_argument('-r',
                        '--reduced',
                        help='reduced output',
                        action='store_true')
    PARSER.add_argument('-i',
                        '--input_dir',
                        help='path to input yaml file and directory')
    PARSER.add_argument('-o',
                        '--output_dir',
                        help='path to output directory')
    PARSER.add_argument('-f',
                        '--format',
                        help='output file format',
                        choices=['hdf5', 'h5', 'csv'],
                        default='csv')
    PARSER.add_argument('-s',
                        '--skip_prompt',
                        help='skip user prompts and warnings',
                        action='store_true')

    ARGS = PARSER.parse_args()
    if ARGS.input_dir:
        u.CONSTANTS['INPUT_FILE'] = ARGS.input_dir
        u.CONSTANTS['INPUT_DIR'] = os.path.dirname(ARGS.input_dir)
    if ARGS.output_dir:
        u.CONSTANTS['OUTPUT_DIR'] = ARGS.output_dir
        if not os.path.exists(ARGS.output_dir):
            os.makedirs(ARGS.output_dir)

    try:
        run(test=ARGS.test, debug=ARGS.debug, file_format=ARGS.format,
            reduced=ARGS.reduced, skip=ARGS.skip_prompt)
    except Exception as err:
        logging.error(err.__repr__())
        print err
        print
