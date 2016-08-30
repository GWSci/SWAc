#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import logging

# Internal modules
from . import io
from . import model as m


###############################################################################
def get_output(data, node):
    """Run the model."""
    logging.info('\tRunning water accounting model for node %d', node)

    for function in [m.get_pefac,              # Column E
                     m.get_canopy_storage,     # Column F
                     m.get_veg_diff,           # Column G
                     m.get_precipitation,      # Column H
                     m.get_snowfall_o,         # Column I
                     m.get_rainfall_o,         # Column J
                     m.get_snow,               # Column K-L
                     m.get_net_rainfall,       # Column M
                     m.get_rawrew,             # Column S
                     m.get_tawrew,             # Column T
                     m.get_ae,                 # Column N-X
                     m.get_unutilized_pe,      # Column Y
                     m.get_perc_through_root,  # Column Z
                     m.get_subroot_leak,       # Column AA
                     m.get_interflow_bypass,   # Column AB
                     m.get_interflow_input,    # Column AC
                     m.get_interflow,          # Column AD-AF
                     m.get_recharge_input,     # Column AG
                     m.get_recharge,           # Column AH-AI
                     m.get_str,                # Column AJ
                     m.get_combined_ae,        # Column AK
                     m.get_evt,                # Column AL
                     m.get_average_in,         # Column AM
                     m.get_average_out,        # Column AN
                     m.get_balance]:           # Column AO

        columns = function(data, node)
        data['output'][node].update(columns)
        logging.info('\t\t"%s()" done', function.__name__)


###############################################################################
def run():
    """Main function."""
    io.start_logging()
    logging.info('Start SWAcMod run')

    data = {}
    data['specs'], data['series'], data['params'] = io.load_params_from_yaml()

    ids = range(1, data['params']['num_nodes'] + 1)
    data['output'] = dict((k, {}) for k in ids)

    io.validate_all(data)
    for node in ids:
        if data['params']['reporting_zone_mapping'][node] == 0:
            continue
        get_output(data, node)
        io.dump_output(data, node)

    logging.info('End SWAcMod run')

    return data


###############################################################################
if __name__ == "__main__":
    run()
