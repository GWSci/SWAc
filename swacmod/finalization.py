#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod validation functions."""

# Standard Library
import re
import logging
import datetime
import multiprocessing

# Third Party Libraries
import numpy as np

# Internal modules
from . import utils as u
import swacmod.feature_flags as ff

try:
    basestring
except NameError:
    basestring = str

def fin_start_date(data, name):
    """Finalize the "start_date" parameter.

    1) if in the right format, convert it to datetime object.
    """
    return _finalize_date(data, name)

def fin_historical_start_date(data, name):
    if (data["params"]["historical_nitrate_process"] != "enabled"):
        return
    _finalize_date(data, name)

def _finalize_date(data, name):
    params = data["params"]

    new_date = str(params[name])
    fields = re.findall(r"^(\d{4})-(\d{2})-(\d{2})$", new_date)
    if not fields:
        msg = ("start_date has to be in the format YYYY-MM-DD "
               + "(e.g. 1980-12-21)")
        raise u.ValidationError(msg)
    params[name] = datetime.datetime(
        int(fields[0][0]), int(fields[0][1]), int(fields[0][2])
    )

def fin_run_name(data, name):
    """Finalize the "run_name" parameter.

    1) if not a string, convert it.
    2) replace non-alphanumeric characters with underscores.
    """
    params = data["params"]
    rnm = params[name]

    if not isinstance(rnm, basestring):
        params[name] = str(rnm)
        logging.info('\t\tConverted "%s" to string', name)

    new_value = re.sub(r"[^a-zA-Z\-0-9]", "_", params[name])
    if new_value != data["params"][name]:
        params[name] = new_value
        logging.info('\t\tNew "%s": %s', name, new_value)

def fin_num_cores(data, name):
    """Finalize the "num_cores" parameter.

    1) if not provided, use the number of cores of the machine.
    """
    count = multiprocessing.cpu_count()
    if data["params"][name] is None:
        data["params"][name] = count
        logging.info('\t\tDefaulted "%s" to %s', name, data["params"][name])
    elif data["params"][name] < 0:
        data["params"][name] = max(count - abs(data["params"][name]), 1)
        logging.info('\t\tSet "%s" to %s', name, data["params"][name])

def fin_disv(data, name):
    """Finalize the "disv" parameter.

    1) if not provided, set it to False.
    """
    if data["params"][name] is None:
        data["params"][name] = False

def fin_output_recharge(data, name):
    """Finalize the "output_recharge" parameter.

    1) if not provided, set it to True.
    """
    if data["params"][name] is None:
        data["params"][name] = True

def fin_output_individual(data, name):
    """Finalize the "output_individual" parameter.

    1) if not provided, set it to "none".
    2) convert it to a string if it's not.
    3) parse it into a set of integers.
    """
    params = data["params"]

    if params[name] is None:
        params[name] = "none"

    oip = str(params[name]).lower()
    sections = [i.strip() for i in oip.split(",")]
    final = []
    for section in sections:
        if section == "all":
            final = range(1, params["num_nodes"] + 1)
            break
        elif section == "none":
            final = []
            break
        if "-" in section:
            try:
                first = int(section.split("-")[0].strip())
                second = int(section.split("-")[1].strip())
                final += range(first, second + 1)
            except (TypeError, ValueError):
                pass
        else:
            try:
                final.append(int(section))
            except (TypeError, ValueError):
                pass

    params[name] = set(final)

def fin_irchcb(data, name):
    """Finalize the "irchcb" parameter.

    1) if not provided, set it to 50.
    """
    if data["params"][name] is None:
        data["params"][name] = 50

def fin_nodes_per_line(data, name):
    """Finalize the "nodes_per_line" parameter.

    1) if not provided, set it to 10.
    """
    if data["params"][name] is None:
        data["params"][name] = 10

def fin_output_fac(data, name):
    """Finalize the "output_fac" parameter.

    1) if not provided, set it to 1.0.
    """
    if data["params"][name] is None:
        data["params"][name] = 1.0

def fin_spatial_output_date(data, name):
    """Finalize the "spatial_output_date" parameter.

    1) if in the right format, convert it to datetime object.
    """
    params = data["params"]
    if params[name] is None:
        return
    if params[name] == "none":
        params[name] = None
        return

    new_date = str(params["spatial_output_date"])
    if new_date != "mean":
        fields = re.findall(r"^(\d{4})-(\d{2})-(\d{2})$", new_date)
        if not fields:
            msg = (
                "spatial_output_date has to be in the format YYYY-MM-DD "
                "(e.g. 1980-01-13)"
            )
            raise u.ValidationError(msg)
        params[name] = datetime.datetime(
            int(fields[0][0]), int(fields[0][1]), int(fields[0][2])
        )

def fin_reporting_zone_mapping(data, name):
    """Finalize the "reporting_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_reporting_zone_names(data, name):
    """Finalize the "reporting_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["reporting_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_rainfall_zone_names(data, name):
    """Finalize the "rainfall_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["rainfall_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_pe_zone_names(data, name):
    """Finalize the "pe_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["pe_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_temperature_zone_mapping(data, name):
    """Finalize the "temperature_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data["params"]
    if params[name] is None:
        nodes = params["num_nodes"]
        params[name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_temperature_zone_names(data, name):
    """Finalize the "temperature_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["temperature_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_tmin_c_zone_mapping(data, name):
    """Finalize the "tmin_c_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data["params"]
    if params[name] is None:
        nodes = params["num_nodes"]
        params[name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_tmin_c_zone_names(data, name):
    """Finalize the "tmin_c_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["tmin_c_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_tmax_c_zone_mapping(data, name):
    """Finalize the "tmax_c_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data["params"]
    if params[name] is None:
        nodes = params["num_nodes"]
        params[name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_tmax_c_zone_names(data, name):
    """Finalize the "tmax_c_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["tmax_c_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_windsp_zone_mapping(data, name):
    """Finalize the "windsp_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data["params"]
    if params[name] is None:
        nodes = params["num_nodes"]
        params[name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_windsp_zone_names(data, name):
    """Finalize the "windsp_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["windsp_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_snow_params_complex(data, name):
    """Finalize the "snow_params_complex" parameter.

    1) if not provided, set it to all [0, 1, 999999, 0].
    """
    if data["params"][name] is None:
        default = [67.55, 0.05, 4.79, 20.0, 100.0, 0.25, 0.95, 1.0, 0.0, 450.0]
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, default)
                                    for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, default)

def fin_subroot_zone_mapping(data, name):
    """Finalize the "subroot_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    params = data["params"]
    if params[name] is None:
        nodes = params["num_nodes"]
        params[name] = dict((k, [1, 1.0]) for k in range(1, nodes + 1))

def fin_subroot_zone_names(data, name):
    """Finalize the "subroot_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        values = [i[0] for i in params["subroot_zone_mapping"].values()]
        zones = len(set(values))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_rapid_runoff_zone_mapping(data, name):
    """Finalize the "rapid_runoff_zone_mapping" parameter.

    1) if not provided, set it to all 0s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_rapid_runoff_zone_names(data, name):
    """Finalize the "rapid_runoff_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["rapid_runoff_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_interflow_zone_mapping(data, name):
    """Finalize the "interflow_zone_mapping" parameter.

    1) if not provided, set it to all 0s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_interflow_zone_names(data, name):
    """Finalize the "interflowf_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["interflow_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_swrecharge_zone_mapping(data, name):
    """Finalize the "swrecharge_zone_mapping" parameter.

    1) if not provided, set it to all 0s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_swrecharge_zone_names(data, name):
    """Finalize the "swrecharge_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["swrecharge_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_single_cell_swrecharge_zone_mapping(data, name):
    """Finalize the "single_cell_swrecharge_zone_mapping" parameter.
    1) if not provided, set it to all 0s.
    """
    if data['params'][name] is None:
        nodes = data['params']['num_nodes']
        data['params'][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_single_cell_swrecharge_zone_names(data, name):
    """Finalize the "single_cell_swrecharge_zone_names" parameter.
    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data['params']
    if params[name] is None:
        zones = len(set(params['single_cell_swrecharge_zone_mapping'].values()))
        params[name] = dict((k, 'Zone%d' % k) for k in range(1, zones + 1))

def fin_single_cell_swrecharge_proportion(data, name):
    """Finalize the "single_cell_swrecharge_proportion" parameter.
    1) if not provided, set it to 0.
    """
    params = data['params']
    zones = data['params']['single_cell_swrecharge_zone_names']
    if params['swrecharge_process'] == 'disabled':
        if params[name] is None:
            params[name] = dict((k, [0.0 for _ in zones]) for k in range(1, 13))
            logging.info('\t\tDefaulted "%s" to [0.0]', name)

        params['ror_prop'] = sorted(params[name].items(), key=lambda x: x[0])
        params['ror_prop'] = np.array([i[1] for i in params['ror_prop']])

def fin_single_cell_swrecharge_limit(data, name):
    """Finalize the "single_cell_swrecharge_limit" parameter.
    1) if not provided, set it to 99999.
    """
    params = data['params']
    zones = data['params']['single_cell_swrecharge_zone_names']

    if params['swrecharge_process'] == 'disabled':
        if params[name] is None:
            params[name] = dict((k, [99999.9 for _ in zones]) for k in
                                range(1, 13))
            logging.info('\t\tDefaulted "%s" to [99999]', name)

        params['ror_limit'] = sorted(params[name].items(), key=lambda x: x[0])
        params['ror_limit'] = np.array([i[1] for i in params['ror_limit']])

def fin_single_cell_swrecharge_activation(data, name):
    """Finalize the "rorecharge_activation" parameter.
    1) if not provided, set it to 0.0.
    """
    params = data['params']
    zones = data['params']['single_cell_swrecharge_zone_names']
    if params[name] is None:
        params[name] = dict((k, [0.0 for _ in zones]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params['ror_act'] = sorted(params[name].items(), key=lambda x: x[0])
    params['ror_act'] = np.array([i[1] for i in params['ror_act']])

def fin_macropore_zone_mapping(data, name):
    """Finalize the "macropore_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_macropore_activation_option(data, name):
    """Finalize the "macropore_activation_option" parameter.

    1) if not provided, set it to SMD.
    """
    if data["params"][name] is None:
        data["params"][name] = 'SMD'

def fin_macropore_zone_names(data, name):
    """Finalize the "macropore_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["macropore_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_soil_zone_names(data, name):
    """Finalize the "macropore_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        try:
            zones = len(list(params["soil_spatial"].items())[0][1])
        except (TypeError, KeyError, IndexError):
            zones = 1
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_landuse_zone_names(data, name):
    """Finalize the "macropore_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        try:
            zones = len(list(params["lu_spatial"].items())[0][1])
        except(TypeError, KeyError, IndexError):
            zones = 1
            params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_canopy_zone_mapping(data, name):
    """Finalize the "canopy_zone_mapping" parameter.

    1) if not provided, set it to all 0s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_canopy_zone_names(data, name):
    """Finalize the "canopy_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["canopy_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_free_throughfall(data, name):
    """Finalize the "free_throughfall" parameter.

    1) if not provided, set it to all 1s.
    """
    if data["params"][name] is None:
        zones = len(data["params"]["canopy_zone_names"])
        default = 1.0
        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_max_canopy_storage(data, name):
    """Finalize the "max_canopy_storage" parameter.

    1) if not provided, set it to all 1s.
    """
    if data["params"][name] is None:
        zones = len(data["params"]["canopy_zone_names"])
        default = 0.0
        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_rapid_runoff_params(data, name):
    """Finalize the "max_canopy_storage" parameter.

    1) if not provided, set it to 0.
    """
    if data["params"][name] is None:
        data["params"][name] = [
            {"class_smd": [0], "class_ri": [0], "values": [[0.0], [0.0]]}
        ]
    else:
        for dataset in data["params"][name]:
            dataset["values"] = [[float(i) for i in row]
                                 for row in dataset["values"]]

def fin_swrecharge_proportion(data, name):
    """Finalize the "swrecharge_proportion" parameter.

    1) if not provided, set it to 0.
    """
    params = data["params"]
    zones = data["params"]["swrecharge_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [0.0 for _ in zones]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params["ror_prop"] = sorted(params[name].items(), key=lambda x: x[0])
    params["ror_prop"] = np.array([i[1] for i in params["ror_prop"]])

def fin_swrecharge_limit(data, name):
    """Finalize the "swrecharge_limit" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["swrecharge_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params["ror_limit"] = sorted(params[name].items(), key=lambda x: x[0])
    params["ror_limit"] = np.array([i[1] for i in params["ror_limit"]])

def fin_macropore_proportion(data, name):
    """Finalize the "macropore_proportion" parameter.

    1) if not provided, set it to 0.
    """
    params = data["params"]
    zones = data["params"]["macropore_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [0.0 for _ in zones]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params["macro_prop"] = sorted(params[name].items(), key=lambda x: x[0])
    params["macro_prop"] = np.array([i[1] for i in params["macro_prop"]])

def fin_macropore_limit(data, name):
    """Finalize the "macropore_limit" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["macropore_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999.9]', name)

    params["macro_limit"] = sorted(params[name].items(), key=lambda x: x[0])
    params["macro_limit"] = np.array([i[1] for i in params["macro_limit"]])

def fin_macropore_activation(data, name):
    """Finalize the "macropore_activation" parameter.

    1) if not provided, set it to 0.0.
    """
    params = data["params"]
    zones = data["params"]["macropore_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [0.0 for _ in zones]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params["macro_act"] = sorted(params[name].items(), key=lambda x: x[0])
    params["macro_act"] = np.array([i[1] for i in params["macro_act"]])

def fin_macropore_recharge(data, name):
    """Finalize the "macropore_recharge" parameter.

    1) if not provided, set it to 0.0.
    """
    params = data["params"]
    zones = data["params"]["macropore_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [0.0 for _ in zones]) for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [0.0]', name)

    params["macro_rec"] = sorted(params[name].items(), key=lambda x: x[0])
    params["macro_rec"] = np.array([i[1] for i in params["macro_rec"]])

def fin_soil_static_params(data, name):
    """Finalize the "soil_static_params" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data["params"]
    if (
        params[name] is None
        and params["fao_process"] == "enabled"
        and params["fao_input"] == "ls"
    ):
        params["fao_process"] = "disabled"
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)

def fin_soil_spatial(data, name):
    """Finalize the "soil_spatial" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data["params"]
    if (
        params[name] is None
        and params["fao_process"] == "enabled"
        and params["fao_input"] == "ls"
    ):
        params["fao_process"] = "disabled"
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)

def fin_lu_spatial(data, name):
    """Finalize the "lu_spatial" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data["params"]
    if params[name] is None and params["fao_process"] == "enabled":
        params["fao_process"] = "disabled"
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)

def fin_zr(data, name):
    """Finalize the "zr" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data["params"]
    if (
        params[name] is None
        and params["fao_process"] == "enabled"
        and params["fao_input"] == "ls"
    ):
        params["fao_process"] = "disabled"
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)

def fin_kc(data, name):
    """Finalize the "kc" parameter.

    1) if not provided, set "fao_process" to "disabled".
    """
    params = data["params"]
    if (
        params[name] is None
        and params["fao_process"] == "enabled"
        and params["fao_input"] == "ls"
    ):
        params["fao_process"] = "disabled"
        logging.info('\t\tSwitched "fao_process" to "disabled", missing %s',
                     name)
    elif params[name]:
        params["kc_list"] = sorted(params[name].items(), key=lambda x: x[0])
        params["kc_list"] = np.array([i[1] for i in params["kc_list"]])

def fin_taw_and_raw(data, name):
    """Finalize the "taw" and "raw" parameters."""

    params = data["params"]
    if params["taw"] is None and params["fao_input"] == "l":
        params["fao_input"] = "ls"
        logging.info('\t\tSwitched "fao_input" to "ls", "taw" is missing')

    if params["raw"] is None and params["fao_input"] == "l":
        params["fao_input"] = "ls"
        logging.info('\t\tSwitched "fao_input" to "ls", "raw" is missing')

    if params["fao_input"] == "ls":
        params["taw"], params["raw"] = u.build_taw_raw(params)
        logging.info('\t\tInferred "taw" and "raw" from soil params')

    elif params["fao_input"] == "l":
        params["taw"] = u.invert_taw_raw(params["taw"], params)
        params["raw"] = u.invert_taw_raw(params["raw"], params)
    if params["taw"] is not None and params["raw"] is not None:
        for node in range(1, params["num_nodes"] + 1):
            params["taw"][node] = np.array(params["taw"][node]).astype(float)
            params["raw"][node] = np.array(params["raw"][node]).astype(float)

def fin_percolation_rejection(data, name):
    """Finalize the "percolation_rejection" parameter.
    1) if not provided, set it to a large number.
    """

    if data["params"][name] is None:
        default = 99999.0
        zones = len(data["params"]["landuse_zone_names"])
        if ff.use_natproc:
            data["params"]["percolation_rejection"] = {"percolation_rejection": [default for _ in range(zones)]}
        else:
            data["params"][name] = [default for _ in range(zones)]
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_percolation_rejection_ts(data, name):
    """Finalize the "percolation_rejection" parameter.

    1) if not provided, set it to a large number.
    """

    series, params = data["series"], data["params"]

    if series[name] is None and params['percolation_rejection_use_timeseries']:
        default = 99999.0
        zones = len(data["params"]["landuse_zone_names"])
        params[name] = np.full([len(series["date"]), zones], default)
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_percolation_rejection_use_timeseries(data, name):
    """Finalize the "percolation_rejection_use_timeseries" parameter.

    1) if not provided, set "percolation_rejection_use_timeseries" to "false".
    """

    params = data["params"]
    if params[name] is None:
        params["percolation_rejection_use_timeseries"] = False
        logging.info('\t\tSwitched "percolation_rejection_use_timeseries" to "false"')

def fin_subsoilzone_leakage_fraction(data, name):
    """Finalize the "subsoilzone_leakage_fraction" parameter.

    1) if not provided, set it to all 0s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        default = 0.0
        data["params"][name] = dict((k, default) for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_init_interflow_store(data, name):
    """Finalize the "init_interflow_store" parameter.
    1) if not provided, set it to zero.
    """
    if data["params"][name] is None:
        default = 0.0
        if data["params"]["interflow_zone_names"] is None:
            zones = 1
        else:
            zones = len(data["params"]["interflow_zone_names"])

        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_interflow_store_bypass(data, name):
    """Finalize the "interflow_store_bypass" parameter.
    1) if not provided, set it to 1.0.
    """
    if data["params"][name] is None:
        default = 1.0
        if data["params"]["interflow_zone_names"] is None:
            zones = 1
        else:
            zones = len(data["params"]["interflow_zone_names"])
        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_infiltration_limit(data, name):
    """Finalize the "infiltration_limit" parameter.
    1) if not provided, set it to 999999.9.
    """
    if data["params"][name] is None:
        default = 999999.9
        if data["params"]["interflow_zone_names"] is None:
            zones = 1
        else:
            zones = len(data["params"]["interflow_zone_names"])
        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_interflow_decay(data, name):
    """Finalize the "interflow_decay" parameter.
    1) if not provided, set it to 0.0.
    """
    if data["params"][name] is None:
        default = 0.0
        if data["params"]["interflow_zone_names"] is None:
            zones = 1
        else:
            zones = len(data["params"]["interflow_zone_names"])
        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_infiltration_limit_ts(data, name):
    """Finalize the "infiltration_limit_ts" parameter.

    1) if not provided, set it to a large number.
    """

    series, params = data["series"], data["params"]

    if series[name] is None and params['infiltration_limit_use_timeseries']:
        default = 99999.0
        zones = len(data["params"]["interflow_zone_names"])
        params[name] = np.full([len(series["date"]), zones], default)
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_infiltration_limit_use_timeseries(data, name):
    """Finalize the "infiltration_limit_use_timeseries" parameter.

    1) if not provided, set "infiltration_limit_use_timeseries" to "false".
    """

    params = data["params"]
    if params[name] is None:
        params["infiltration_limit_use_timeseries"] = False
        logging.info('\t\tSwitched "infiltration_limit_use_timeseries" to "false"')

def fin_interflow_decay_ts(data, name):
    """Finalize the "interflow_decay_ts" parameter.

    1) if not provided, set it to a large number.
    """

    series, params = data["series"], data["params"]

    if series[name] is None and params['interflow_decay_use_timeseries']:
        default = 99999.0
        zones = len(data["params"]["interflow_zone_names"])
        params[name] = np.full([len(series["date"]), zones], default)
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

def fin_interflow_decay_use_timeseries(data, name):
    """Finalize the "interflow_decay_use_timeseries" parameter.

    1) if not provided, set "interflow_decay_use_timeseries" to "false".
    """

    params = data["params"]
    if params[name] is None:
        params["interflow_decay_use_timeseries"] = False
        logging.info('\t\tSwitched "interflow_decay_use_ts" to "false"')

def fin_recharge_attenuation_params(data, name):
    """Finalize the "recharge_attenuation_params" parameter.

    1) if not provided, set it to all [0, 1, 999999].
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, [0, 1, 999999])
                                    for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, [0, 1, 999999])

def fin_sw_params(data, name):
    """Finalize the "sw_params" parameter.

    1) if not provided, set it to all [0.0, 1.0].
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, [0.0, 1.0])
                                    for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, [0.0, 1.0])

def fin_sw_zone_mapping(data, name):
    """Finalize the "sw_zone_mapping" parameter.

    1) if not provided, set it to all 1s.
    """
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        data["params"][name] = dict((k, 1) for k in range(1, nodes + 1))

def fin_sw_zone_names(data, name):
    """Finalize the "sw_zone_names" parameter.

    1) if not provided, set it to "Zone1", "Zone2" etc.
    """
    params = data["params"]
    if params[name] is None:
        zones = len(set(params["reporting_zone_mapping"].values()))
        params[name] = dict((k, "Zone%d" % k) for k in range(1, zones + 1))

def fin_sw_init_ponding(data, name):
    """Finalize the "sw_init_ponding" parameter.

    1) if not provided, set it to 0.0.
    """
    params = data["params"]
    if params[name] is None:
        params[name] = 0.0

def fin_sw_ponding_area(data, name):
    """Finalize the "ponding_area" parameter.
    1) if not provided, set it to 0.0.
    """

    params = data["params"]
    if data["params"][name] is None:
        default = 0.0
        if data["params"]["sw_zone_names"] is None:
            zones = 1
        else:
            zones = len(data["params"]["sw_zone_names"])
        data["params"][name] = {zone: default for zone in range(1, zones + 1)}
        logging.info('\t\tDefaulted "%s" to %.2f', name, default)

    params["sw_pond_area"] = sorted(params[name].items(), key=lambda x: x[0])
    params["sw_pond_area"] = np.array([i[1]
                                       for i in params["sw_pond_area"]])

def fin_sw_pe_to_open_water(data, name):
    """Finalize the "sw_pe_to_open_water" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["sw_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params["sw_pe_to_open_wat"] = sorted(params[name].items(), key=lambda x: x[0])
    params["sw_pe_to_open_wat"] = np.array([i[1]
                                            for i in params["sw_pe_to_open_wat"]])

def fin_sw_direct_recharge(data, name):
    """Finalize the "sw_direct_recharge" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["sw_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params["sw_direct_rech"] = sorted(params[name].items(), key=lambda x: x[0])
    params["sw_direct_rech"] = np.array([i[1] for i in params["sw_direct_rech"]])

def fin_sw_activation(data, name):
    """Finalize the "sw_activation" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["sw_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params["sw_activ"] = sorted(params[name].items(), key=lambda x: x[0])
    params["sw_activ"] = np.array([i[1] for i in params["sw_activ"]])

def fin_sw_bed_infiltration(data, name):
    """Finalize the "sw_bed_infiltration" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["sw_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params["sw_bed_infiltn"] = sorted(params[name].items(), key=lambda x: x[0])
    params["sw_bed_infiltn"] = np.array([i[1] for i in params["sw_bed_infiltn"]])

def fin_sw_downstream(data, name):
    """Finalize the "sw_downstream" parameter.

    1) if not provided, set it to 99999.
    """
    params = data["params"]
    zones = data["params"]["sw_zone_names"]
    if params[name] is None:
        params[name] = dict((k, [99999.9 for _ in zones])
                            for k in range(1, 13))
        logging.info('\t\tDefaulted "%s" to [99999]', name)

    params["sw_downstr"] = sorted(params[name].items(), key=lambda x: x[0])
    params["sw_downstr"] = np.array([i[1] for i in params["sw_downstr"]])

def fin_output_sfr(data, name):
    """Finalize the "output_sfr" parameter.

    1) if not provided, set "output_sfr" to "false".
    """

    params = data["params"]
    if params[name] is None or params[name] is None:
        params["output_sfr"] = False
        logging.info('\t\tSwitched "output_sfr" to "false"')

def fin_attenuate_sfr_flows(data, name):
    if name not in data["params"]:
        data["params"][name] = False

def fin_sfr_obs(data, name):
    """Finalize the "sfr_obs" parameter.

    1) if not provided, set it to ''.
    """
    if data["params"][name] is None:
        data["params"][name] = ''

def fin_istcb1(data, name):
    """Finalize the "istcb1" parameter.

    1) if not provided, set it to 50.
    """
    if data["params"][name] is None:
        data["params"][name] = 50

def fin_istcb2(data, name):
    """Finalize the "istcb2" parameter.

    1) if not provided, set it to 55.
    """
    if data["params"][name] is None:
        data["params"][name] = 55

def fin_routing_process(data, name):
    """Finalize the "routing_process" parameter.

    1) if not provided, set to "disabled".
    """
    params = data["params"]
    if params[name] is None:
        params["routing_process"] = "disabled"
        logging.info('\t\tSwitched "routing_process" to "disabled"')
        params["output_sfr"] = False
        logging.info('\t\tSwitched "output_sfr" to "false", missing %s', name)

def fin_routing_topology(data, name):
    """Finalize the "routing_topology" parameter.

    1) if not provided, set it to all zero.
    """
    params = data["params"]
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        zeros = [0] * 3 + [0.0] * 7
        data["params"][name] = dict((k, zeros) for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, zeros)
        # params['output_sfr'] = 'false'
        params["output_sfr"] = False
        logging.info('\t\tSwitched "output_sfr" to "false", missing %s', name)
        params["routing_process"] = "disabled"
        logging.info('\t\tSwitch "routing_process" to "disabled", missing %s',
                     name)

def fin_swdis_locs(data, name):
    """Finalize the "swdis_locs" parameter.

    1) if not provided, set it to all zeros.
    """
    params = data["params"]
    if params[name] is None:
        data["params"][name] = {0: 0}

def fin_swabs_locs(data, name):
    """Finalize the "swabs_locs" parameter.

    1) if not provided, set it to all zeros.
    """
    params = data["params"]
    if params[name] is None:
        data["params"][name] = {0: 0}

def fin_swdis_f(data, name):
    """Finalize the "swdis_f" parameter.

    1) if not provided, set it to 0
    """
    if data["params"][name] is None:
        data["params"][name] = 0

def fin_swabs_f(data, name):
    """Finalize the "swabs_f" parameter.

    1) if not provided, set it to 0
    """
    if data["params"][name] is None:
        data["params"][name] = 0

def fin_date(data, name):
    """Finalize the "date" series."""
    series, params = data["series"], data["params"]
    time_periods = params["time_periods"]
    start_date = params["start_date"]
    series[name] = _fin_date_series(time_periods, start_date)

def fin_months(data, name):
    series = data["series"]
    dates = np.array([np.datetime64(str(i.date())) for i in series["date"]])
    series[name] = dates.astype("datetime64[M]").astype(int) % 12

def fin_historical_nitrate_days(data, name):
    if (data["params"]["historical_nitrate_process"] != "enabled"):
        return
    series, params = data["series"], data["params"]
    time_periods = params["historical_time_periods"]
    start_date = params["historical_start_date"]
    series[name] = _fin_date_series(time_periods, start_date)

def _fin_date_series(time_periods, start_date):
    max_time = max([i for j in time_periods for i in j]) - 1
    day = datetime.timedelta(1)
    return [start_date + day * num for num in range(max_time)]

def fin_rainfall_ts(data, name):
    """Finalize the "rainfall_ts" series."""
    series = data["series"]
    series[name] = np.array(series[name])

def fin_swdis_ts(data, name):
    """Finalize the "swdis_ts" series."""

    series = data["series"]
    if series[name] is None:
        series[name] = [0.0]
    series[name] = np.array(series[name])

def fin_swabs_ts(data, name):
    """Finalize the "swabs_ts" series."""

    series = data["series"]
    if series[name] is None:
        series[name] = [0.0]

    series[name] = np.array(series[name])

def fin_pe_ts(data, name):
    """Finalize the "pe_ts" series."""
    series, specs, params = data["series"], data["specs"], data["params"]

    fao = params["fao_process"]
    canopy = params["canopy_process"]
    if fao != "enabled" and canopy != "enabled":
        zones = len(set(params["pe_zone_mapping"].values()))
        series[name] = np.zeros([len(series["date"]), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)
    elif not specs[name]["required"]:
        specs[name]["required"] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)

def fin_temperature_ts(data, name):
    """Finalize the "temperature_ts" series."""
    series, specs, params = data["series"], data["specs"], data["params"]

    if params["snow_process_simple"] == "enabled" and not specs[name]["required"]:
        specs[name]["required"] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        zones = len(set(params["temperature_zone_mapping"].values()))
        series[name] = np.zeros([len(series["date"]), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)

def fin_windsp_ts(data, name):
    """Finalize the "winsp_ts" series."""
    series, specs, params = data["series"], data["specs"], data["params"]

    if params["snow_process_complex"] == "enabled" and not specs[name]["required"]:
        specs[name]["required"] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        zones = len(set(params["windsp_zone_mapping"].values()))
        series[name] = np.zeros([len(series["date"]), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)

def fin_tmax_c_ts(data, name):
    """Finalize the "temperature_ts" series."""
    series, specs, params = data["series"], data["specs"], data["params"]

    if params["snow_process_complex"] == "enabled" and not specs[name]["required"]:
        specs[name]["required"] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        zones = len(set(params["tmax_c_zone_mapping"].values()))
        series[name] = np.zeros([len(series["date"]), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)

def fin_tmin_c_ts(data, name):
    """Finalize the "temperature_ts" series."""
    series, specs, params = data["series"], data["specs"], data["params"]

    if params["snow_process_complex"] == "enabled" and not specs[name]["required"]:
        specs[name]["required"] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        zones = len(set(params["tmin_c_zone_mapping"].values()))
        series[name] = np.zeros([len(series["date"]), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)

def fin_subroot_leakage_ts(data, name):
    """Finalize the "subroot_leakage_ts" series."""
    series, specs, params = data["series"], data["specs"], data["params"]

    if params["leakage_process"] == "enabled" and not specs[name]["required"]:
        specs[name]["required"] = True
        series[name] = np.array(series[name])
        logging.info('\t\tSwitched "%s" to "required"', name)
    else:
        values = [i[0] for i in params["subroot_zone_mapping"].values()]
        zones = len(set(values))
        series[name] = np.zeros([len(series["date"]), zones])
        logging.info('\t\tDefaulted "%s" to 0.0', name)

def fin_output_evt(data, name):
    """Finalize the "output_evt" parameter.

    1) if not provided, set it to False.
    """
    if data["params"][name] is None:
        data["params"][name] = False

def fin_excess_sw_process(data, name):
    """Finalize the "excess_sw_process" parameter.

    1) if not provided, set "excess_sw_process" to "disabled".
    """
    params = data["params"]
    if params[name] is None:
        params["excess_sw_process"] = "disabled"
        logging.info('\t\tSwitch "excess_sw_proc" to "disabled", missing %s',
                     name)

def fin_gwmodel_type(data, name):
    """Finalize the "gwmodel_type" parameter.

    1) if not provided, set "gwmodel_type" to "mfusg".
    """
    params = data["params"]
    if params[name] is None:
        params["gwmodel_type"] = "mfusg"
        logging.info('\t\tSwitched "gwmodel_type" to "mfusg", missing %s',
                     name)

def fin_evt_parameters(data, name):

    """Finalize the "evt_parameters" parameter.
    1) if not provided, set it to all zero.
    """

    params = data["params"]
    if data["params"][name] is None:
        nodes = data["params"]["num_nodes"]
        zeros = [0] + [0.0] * 2
        data["params"][name] = dict((k, zeros) for k in range(1, nodes + 1))
        logging.info('\t\tDefaulted "%s" to %s', name, zeros)
        params["output_evt"] = False
        logging.info('\t\tSwitched "output_evt" to "false", missing %s', name)

    if (params["output_sfr"] and params["excess_sw_process"] != "disabled"):
        msg = ("Modflow SFR output and excess_sw_process enabled"
               " - you may not want to do this")
        logging.info('\t\t' + msg)
        print("Warning: " + msg)

def fin_ievtcb(data, name):
    """Finalize the "ievtcb" parameter.

    1) if not provided, set it to 50.
    """
    if data["params"][name] is None:
        data["params"][name] = 50

def fin_nevtopt(data, name):
    """Finalize the "nevtopt" parameter.

    1) if not provided, set it to 2.
    """
    if data["params"][name] is None:
        data["params"][name] = 2

FUNC_PARAMS = [
    fin_start_date,
    fin_historical_start_date,
    fin_run_name,
    fin_num_cores,
    fin_output_recharge,
    fin_output_individual,
    fin_irchcb,
    fin_nodes_per_line,
    fin_output_fac,
    fin_spatial_output_date,
    fin_reporting_zone_mapping,
    fin_reporting_zone_names,
    fin_rainfall_zone_names,
    fin_pe_zone_names,
    fin_temperature_zone_mapping,
    fin_temperature_zone_names,
    fin_tmax_c_zone_mapping,
    fin_tmax_c_zone_names,
    fin_tmin_c_zone_mapping,
    fin_tmin_c_zone_names,
    fin_windsp_zone_mapping,
    fin_windsp_zone_names,
    fin_snow_params_complex,
    fin_subroot_zone_mapping,
    fin_subroot_zone_names,
    fin_rapid_runoff_zone_mapping,
    fin_rapid_runoff_zone_names,
    fin_interflow_zone_mapping,
    fin_interflow_zone_names,
    fin_interflow_decay,
    fin_interflow_store_bypass,
    fin_infiltration_limit,
    fin_init_interflow_store,
    fin_infiltration_limit_use_timeseries,
    fin_interflow_decay_use_timeseries,
    fin_swrecharge_zone_mapping,
    fin_swrecharge_zone_names,
    fin_single_cell_swrecharge_zone_mapping,
    fin_single_cell_swrecharge_zone_names,
    fin_macropore_zone_mapping,
    fin_macropore_zone_names,
    fin_soil_zone_names,
    fin_landuse_zone_names,
    fin_canopy_zone_mapping,
    fin_canopy_zone_names,
    fin_free_throughfall,
    fin_max_canopy_storage,
    fin_rapid_runoff_params,
    fin_swrecharge_proportion,
    fin_swrecharge_limit,
    fin_single_cell_swrecharge_proportion,
    fin_single_cell_swrecharge_limit,
    fin_single_cell_swrecharge_activation,
    fin_macropore_proportion,
    fin_macropore_limit,
    fin_macropore_activation,
    fin_macropore_recharge,
    fin_macropore_activation_option,
    fin_soil_static_params,
    fin_soil_spatial,
    fin_lu_spatial,
    fin_taw_and_raw,
    fin_zr,
    fin_kc,
    fin_percolation_rejection,
    fin_percolation_rejection_use_timeseries,
    fin_subsoilzone_leakage_fraction,
    fin_recharge_attenuation_params,
    fin_sw_params,
    fin_sw_zone_mapping,
    fin_sw_zone_names,
    fin_sw_pe_to_open_water,
    fin_sw_direct_recharge,
    fin_sw_activation,
    fin_sw_bed_infiltration,
    fin_sw_downstream,
    fin_sw_init_ponding,
    fin_sw_ponding_area,
    fin_output_sfr,
    fin_attenuate_sfr_flows,
    fin_sfr_obs,
    fin_istcb1,
    fin_istcb2,
    fin_routing_process,
    fin_routing_topology,
    fin_swdis_locs,
    fin_swabs_locs,
    fin_swdis_f,
    fin_swabs_f,
    fin_output_evt,
    fin_excess_sw_process,
    fin_evt_parameters,
    fin_ievtcb,
    fin_nevtopt,
    fin_gwmodel_type,
    fin_disv
]

FUNC_SERIES = [
    fin_date,
    fin_months,
    fin_historical_nitrate_days,
    fin_rainfall_ts,
    fin_pe_ts,
    fin_temperature_ts,
    fin_tmax_c_ts,
    fin_tmin_c_ts,
    fin_windsp_ts,
    fin_subroot_leakage_ts,
    fin_swabs_ts,
    fin_swdis_ts,
    fin_percolation_rejection_ts,
    fin_infiltration_limit_ts,
    fin_interflow_decay_ts,
]

def finalize_params(data):
    """Finalize all parameters."""
    logging.info("\tFinalizing parameters")

    for function in FUNC_PARAMS:
        param = function.__name__.replace("fin_", "")
        logging.debug('\t\t"%s" ww final', param)
        try:
            function(data, param)
        except Exception as err:
            raise u.FinalizationError(
                'Could not finalize "%s": %s' % (param, err.__repr__())
            )
        logging.debug('\t\t"%s" finalized', param)

    logging.info("\tDone.")

def finalize_series(data):
    """Finalize all time series."""
    logging.info("\tFinalizing time series")

    for function in FUNC_SERIES:
        series = function.__name__.replace("fin_", "")
        try:
            function(data, series)
        except Exception as err:
            raise u.FinalizationError(
                'Could not finalize "%s": %s' % (series, err.__repr__())
            )
        logging.debug('\t\t"%s" finalized', series)

    logging.info("\tDone.")
