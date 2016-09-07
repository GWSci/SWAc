#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import sys
import time
import logging
import multiprocessing

# Third Party Libraries
import numpy as np

# Internal modules
from . import io
from . import utils as u
from . import model as m


###############################################################################
def get_output(data, node):
    """Run the model."""
    logging.debug('\tRunning model for node %d', node)

    start = time.time()

    data['output'] = {}
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

        columns = function(data, node)
        data['output'].update(columns)
        logging.debug('\t\t"%s()" done', function.__name__)

    end = time.time()
    logging.info('\tNode %d done (%dms).', node, (end - start) * 1000)


###############################################################################
def run_process(num, ids, data, test):
    """Run model for a chunk of nodes."""
    logging.info('Process %d started', num)
    for node in ids:
        if data['params']['reporting_zone_mapping'][node] == 0:
            continue
        get_output(data, node)
        logging.debug('RAM usage is %.2fMb', u.get_ram_usage_for_process())
        if not test:
            io.dump_output(data, node)
    logging.info('Process %d ended', num)


###############################################################################
def run(test=False):
    """Run model for all nodes."""
    io.start_logging()
    logging.info('Start SWAcMod run')

    data = {}
    if test:
        input_file = u.CONSTANTS['TEST_INPUT_FILE']
        input_dir = u.CONSTANTS['TEST_INPUT_DIR']
        specs, series, params = io.load_params_from_yaml(input_file=input_file,
                                                         input_dir=input_dir)
    else:
        specs, series, params = io.load_params_from_yaml()

    if specs is None:
        print
        sys.exit()

    data['specs'], data['series'], data['params'] = specs, series, params
    io.validate_all(data)

    ids = range(1, data['params']['num_nodes'] + 1)
    chunks = np.array_split(ids, data['params']['num_cores'])

    procs = {}
    for num, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        procs[num] = multiprocessing.Process(target=run_process,
                                             args=(num, chunk, data, test, ))
        procs[num].start()

    while all(i.is_alive() for i in procs.values()):
        time.sleep(0.5)

    logging.info('End SWAcMod run')

    return data


###############################################################################
if __name__ == "__main__":
    try:
        ARG = (True if sys.argv[1] == 'test' else False)
    except IndexError:
        ARG = False
    run(test=ARG)
