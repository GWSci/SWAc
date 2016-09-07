#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import sys
import time
import logging

# Internal modules
from . import io
from . import utils as u
from . import model as m


###############################################################################
def get_output(data, node):
    """Run the model."""
    logging.info('\tRunning model for node %d', node)

    start = time.time()

    data['output'] = {}
    for function in [m.get_pefac,                  # Column E
                     m.get_canopy_storage,         # Column F
                     m.get_net_pefac,              # Column G
                     m.get_precip_to_ground,       # Column H
                     m.get_snowfall_o,             # Column I
                     m.get_rainfall_o,             # Column J
                     m.get_snow,                   # Column K-L
                     m.get_net_rainfall,           # Column M
                     m.get_rawrew,                 # Column S
                     m.get_tawrew,                 # Column T
                     m.get_ae,                     # Column N-X
                     m.get_unutilised_pe,          # Column Y
                     m.get_perc_through_root,      # Column Z
                     m.get_subroot_leak,           # Column AA
                     m.get_interflow_bypass,       # Column AB
                     m.get_interflow_store_input,  # Column AC
                     m.get_interflow,              # Column AD-AF
                     m.get_recharge_store_input,   # Column AG
                     m.get_recharge,               # Column AH-AI
                     m.get_combined_str,           # Column AJ
                     m.get_combined_ae,            # Column AK
                     m.get_evt,                    # Column AL
                     m.get_average_in,             # Column AM
                     m.get_average_out,            # Column AN
                     m.get_balance]:               # Column AO

        columns = function(data, node)
        data['output'].update(columns)
        logging.debug('\t\t"%s()" done', function.__name__)

    end = time.time()
    logging.info('\tDone (%dms).', (end - start) * 1000)


###############################################################################
def run(test=False):
    """Main function."""
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
    ids = range(1, data['params']['num_nodes'] + 1)

    io.validate_all(data)
    for node in ids:
        if data['params']['reporting_zone_mapping'][node] == 0:
            continue
        get_output(data, node)
        logging.info('RAM usage is %.2fMb', u.get_ram_usage_for_process())
        if not test:
            io.dump_output(data, node)

    logging.info('End SWAcMod run')

    return data


###############################################################################
if __name__ == "__main__":
    try:
        ARG = (True if sys.argv[1] == 'test' else False)
    except IndexError:
        ARG = False
    run(test=ARG)
