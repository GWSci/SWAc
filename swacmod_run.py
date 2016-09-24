#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import os
import sys
import time
import logging
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
                     m.get_balance]:

        columns = function(data, output, node)
        output.update(columns)
        logging.debug('\t\t"%s()" done', function.__name__)

    end = time.time()
    logging.debug('\tNode %d done (%dms).', node, (end - start) * 1000)
    return output


###############################################################################
def run_process(num, ids, data, test, reporting, recharge, log_path):
    """Run model for a chunk of nodes."""
    io.start_logging(path=log_path)
    logging.info('Process %d started (%d nodes, test is %s)',
                 num, len(ids), test)
    for node in ids:
        rep_zone = data['params']['reporting_zone_mapping'][node]
        if rep_zone == 0:
            continue
        output = get_output(data, node)
        logging.debug('RAM usage is %.2fMb', u.get_ram_usage_for_process())
        if not test:
            if node in data['params']['output_individual']:
                io.dump_water_balance(data, output, node=node)
            if rep_zone not in reporting:
                reporting[rep_zone] = output.copy()
            else:
                for key in output:
                    reporting[rep_zone][key] += output[key]
            recharge[node] = output['combined_recharge'].copy()
    logging.info('Process %d ended', num)


###############################################################################
def run(test=False):
    """Run model for all nodes."""
    log_path = io.start_logging()
    logging.info('Start SWAcMod run')

    manager = Manager()
    reporting = manager.dict()
    recharge = manager.dict()

    if test:
        input_file = u.CONSTANTS['TEST_INPUT_FILE']
        input_dir = u.CONSTANTS['TEST_INPUT_DIR']
        data = io.load_params_from_yaml(input_file=input_file,
                                        input_dir=input_dir)
    else:
        data = io.load_params_from_yaml()

    if data['specs'] is None:
        print
        sys.exit()

    io.validate_all(data)
    ids = range(1, data['params']['num_nodes'] + 1)
    chunks = np.array_split(ids, data['params']['num_cores'])

    procs = {}
    for num, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        procs[num] = Process(target=run_process,
                             args=(num, chunk, data, test, reporting,
                                   recharge, log_path))
        procs[num].start()

    for num in procs:
        procs[num].join()

    if not test:
        for key in reporting.keys():
            io.dump_water_balance(data, reporting[key], zone=key)
        if data['params']['output_recharge']:
            io.dump_recharge_file(data, recharge)

    logging.info('End SWAcMod run')


###############################################################################
if __name__ == "__main__":
    try:
        ARG = (True if sys.argv[1] == 'test' else False)
    except IndexError:
        ARG = False
    run(test=ARG)
