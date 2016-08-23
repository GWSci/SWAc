#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod main."""

# Standard Library
import logging

# Internal modules
from . import io
from . import model as m


###############################################################################
def get_output(data):
    """Run the model."""
    logging.info('\tRunning water accounting model')

    data['output'] = {}

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

        columns = function(data)
        data['output'].update(columns)
        logging.info('\t\t"%s()" done', function.__name__)

    logging.info('\tWater accounting model done')


###############################################################################
def run():
    """Main function."""
    io.start_logging()

    logging.info('Start SWAcMod run')
    data = {}
    data['series'] = io.load_input_from_excel()
    data['params'] = io.load_params_from_excel()
    get_output(data)
    io.dump_output(data)
    logging.info('End SWAcMod run')

    return data


###############################################################################
if __name__ == "__main__":
    run()
