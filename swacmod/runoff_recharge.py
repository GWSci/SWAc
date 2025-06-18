from __future__ import print_function
import swacmod.feature_flags as ff
import numpy as np
from tqdm import tqdm
from swacmod import utils as u
from swacmod import model as m

def calculate_runoff_recharge(data, days, nnodes, params, recharge, recharge_aggregate, runoff, runoff_aggregate, runoff_recharge_aggregate, stuff):
    if params["swrecharge_process"] == "enabled":

        for catchment in data["params"]["reporting_zone_mapping"].values():
            stuff.reporting_agg2[catchment] = {}

        runoff_recharge = _make_runoff_recharge(recharge, runoff)
        runoff, recharge = m.do_swrecharge_mask(data, runoff, recharge)
        runoff_recharge = _get_runoff_recharge_for_cat_output_purposes(recharge, runoff, runoff_recharge)

        for node in tqdm(list(m.all_days_mask(data).nodes()), desc="Aggregating Fluxes      "):
            # get indices of output for this node
            idx = range(node, (nnodes * days) + 1, nnodes)
            pond_area = _lookup_pond_area(data, node, params)
            recharge_array = _calculate_recharge_aggregate(data, idx, nnodes, node, recharge, recharge_aggregate)
            runoff_array = _calculate_runoff_aggregate(data, idx, nnodes, node, runoff, runoff_aggregate)
            runoff_recharge_array = _calculate_runoff_recharge_aggregate(data, idx, nnodes, node, runoff_recharge, runoff_recharge_aggregate)
            _amend_catchment_output_values(data, node, pond_area, runoff_recharge_array, stuff)
            _extract_node_for_output_individual(data, node, recharge_array, runoff_array, runoff_recharge_array, stuff)

        _copy_new_bits_into_cat_output(stuff)

def _make_runoff_recharge(recharge, runoff):
    if ff.disable_multiprocessing:
        if ff.use_natproc:
            runoff_recharge = recharge.copy()
        else:
            runoff_recharge = runoff.copy()
    else:
        if ff.use_natproc:
            runoff_recharge = np.frombuffer(recharge.get_obj(),
                                            dtype=np.float32).copy()
        else:
            runoff_recharge = np.frombuffer(runoff.get_obj(),
                                            dtype=np.float32).copy()
    return runoff_recharge

def _get_runoff_recharge_for_cat_output_purposes(recharge, runoff, runoff_recharge):
    if ff.use_natproc:
        if ff.disable_multiprocessing:
            runoff_recharge = recharge
        else:
            runoff_recharge = np.frombuffer(recharge.get_obj(),
                                            dtype=np.float32) - runoff_recharge
    else:
        if ff.disable_multiprocessing:
            runoff_recharge -= runoff
        else:
            runoff_recharge -= np.frombuffer(runoff.get_obj(),
                                             dtype=np.float32)
    return runoff_recharge

def _calculate_recharge_aggregate(data, idx, nnodes, node, recharge, recharge_aggregate):
    recharge_array = _extract_array_for_node(idx, recharge)
    recharge_aggregate_for_node = u.aggregate_array(data, recharge_array)
    for period, val in enumerate(recharge_aggregate_for_node):
        recharge_aggregate[(nnodes * period) + int(node)] = val
    return recharge_array

def _calculate_runoff_aggregate(data, idx, nnodes, node, runoff, runoff_aggregate):
    runoff_array = _extract_array_for_node(idx, runoff)
    runoff_aggregate_for_node = u.aggregate_array(data, runoff_array)
    for period, val in enumerate(runoff_aggregate_for_node):
        runoff_aggregate[(nnodes * period) + int(node)] = val
    return runoff_array

def _calculate_runoff_recharge_aggregate(data, idx, nnodes, node, runoff_recharge, runoff_recharge_aggregate):
    runoff_recharge_array = _extract_array_for_node_from_list(idx, runoff_recharge)
    runoff_recharge_aggregate_for_node = u.aggregate_array(data, runoff_recharge_array)
    for period, val in enumerate(runoff_recharge_aggregate_for_node):
        runoff_recharge_aggregate[(nnodes * period) + int(node)] = val
    return runoff_recharge_array

def _extract_node_for_output_individual(data, node, rch_array, ro_array, ror_array, stuff):
    if node in data["params"]["output_individual"]:
        # amend single_node_output with ror values
        # this method required due to upstream bug
        tmp_node = stuff.single_node_output[node]
        tmp_node["runoff_recharge"] = ror_array.copy()
        tmp_node["combined_recharge"] = np.copy(rch_array)
        tmp_node["combined_str"] = np.copy(ro_array)
        stuff.single_node_output[node] = tmp_node

def _amend_catchment_output_values(data, node, pond_area, ror_array, stuff):
    rep_zone = data["params"]["reporting_zone_mapping"][node]
    if _do_we_do_this_bit(rep_zone):
        area = data["params"]["node_areas"][node]
        ror = {"runoff_recharge": ror_array}

        if "runoff_recharge" not in stuff.reporting_agg2[rep_zone]:
            reporting = None
        else:
            reporting = stuff.reporting_agg2[rep_zone]["runoff_recharge"]

        stuff.reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(
            ror, area, pond_area, reporting=reporting)

def _do_we_do_this_bit(rep_zone):
    if ff.use_natproc:
        do_this_bit = rep_zone > 0
    else:
        do_this_bit = True
    return do_this_bit

def _extract_array_for_node(idx, source_array):
    if ff.disable_multiprocessing:
        tmp = source_array
    else:
        tmp = np.frombuffer(source_array.get_obj(), dtype=np.float32)
    return _extract_array_for_node_from_list(idx, tmp)

def _extract_array_for_node_from_list(idx, source_array):
    return np.array(source_array[idx], dtype=np.float64, copy=True)

def _lookup_pond_area(data, node, params):
    if params['sw_ponding_process'] == 'enabled':
        zone_sw = data['params']['sw_zone_mapping'][node]
        pond_area = data['params']['sw_ponding_area'][zone_sw]
    else:
        pond_area = 0.0
    return pond_area

def _copy_new_bits_into_cat_output(stuff):
    term = "runoff_recharge"
    for catchment in stuff.reporting_agg2:
        if "runoff_recharge" in stuff.reporting_agg2[catchment]:
            stuff.reporting_agg[catchment]["combined_recharge"] += stuff.reporting_agg2[
                catchment][term][term]
            stuff.reporting_agg[catchment]["combined_str"] -= stuff.reporting_agg2[catchment][
                term][term]
            stuff.reporting_agg[catchment]["runoff_recharge"] = stuff.reporting_agg2[
                catchment][term][term]
