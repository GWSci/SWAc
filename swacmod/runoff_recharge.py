from __future__ import print_function
import swacmod.feature_flags as ff
import numpy as np
from tqdm import tqdm
from swacmod import utils as u
from swacmod import model as m

def calculate_runoff_recharge(data, days, nnodes, params, recharge, recharge_agg, runoff, runoff_agg, runoff_recharge_agg, stuff):
    if params["swrecharge_process"] == "enabled":

        for cat in data["params"]["reporting_zone_mapping"].values():
            stuff.reporting_agg2[cat] = {}

        runoff_recharge = _make_runoff_recharge(recharge, runoff)
        runoff, recharge = m.do_swrecharge_mask(data, runoff, recharge)
        runoff_recharge = _get_runoff_recharge_for_cat_output_purposes(recharge, runoff, runoff_recharge)

        for node in tqdm(list(m.all_days_mask(data).nodes()), desc="Aggregating Fluxes      "):
            _aggregate_amended_recharge_and_runoff_arrays_by_output_periods(data, days, nnodes, node, params, recharge, recharge_agg, runoff, runoff_agg, runoff_recharge, runoff_recharge_agg, stuff)

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

def _aggregate_amended_recharge_and_runoff_arrays_by_output_periods(data, days, nnodes, node, params, recharge, recharge_agg, runoff, runoff_agg, runoff_recharge, runoff_recharge_agg, stuff):
    # get indices of output for this node
    idx = range(node, (nnodes * days) + 1, nnodes)
    pond_area = _lookup_pond_area(data, node, params)
    rch_array = _extract_array_for_node(idx, recharge)
    ro_array = _extract_array_for_node(idx, runoff)
    ror_array = np.array(runoff_recharge[idx],
                         dtype=np.float64,
                         copy=True)
    # aggregate single node of recharge array
    rch_agg = u.aggregate_array(data, rch_array)
    # aggregate single node of runoff array
    ro_agg = u.aggregate_array(data, ro_array)
    # aggregate single node of runoff rech array
    ror_agg = u.aggregate_array(data, ror_array)
    for period, val in enumerate(rch_agg):
        recharge_agg[(nnodes * period) + int(node)] = val
    for period, val in enumerate(ro_agg):
        runoff_agg[(nnodes * period) + int(node)] = val
    for period, val in enumerate(ror_agg):
        runoff_recharge_agg[(nnodes * period) + int(node)] = val
    # amend catchment output values
    rep_zone = data["params"]["reporting_zone_mapping"][node]
    if ff.use_natproc:
        do_this_bit = rep_zone > 0
    else:
        do_this_bit = True
    if do_this_bit:
        area = data["params"]["node_areas"][node]
        ror = {"runoff_recharge": ror_array}

        if "runoff_recharge" not in stuff.reporting_agg2[rep_zone]:
            stuff.reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(
                ror, area, pond_area)
        else:
            stuff.reporting_agg2[rep_zone]["runoff_recharge"] = m.aggregate(
                ror,
                area,
                pond_area,
                reporting=stuff.reporting_agg2[rep_zone]["runoff_recharge"])
    # check for single node
    if node in data["params"]["output_individual"]:
        # amend single_node_output with ror values
        # this method required due to upstream bug
        tmp_node = stuff.single_node_output[node]
        tmp_node["runoff_recharge"] = ror_array.copy()
        tmp_node["combined_recharge"] = np.copy(rch_array)
        tmp_node["combined_str"] = np.copy(ro_array)
        stuff.single_node_output[node] = tmp_node

def _extract_array_for_node(idx, source_array):
    if ff.disable_multiprocessing:
        tmp = source_array
    else:
        tmp = np.frombuffer(source_array.get_obj(), dtype=np.float32)
    return np.array(tmp[idx], dtype=np.float64, copy=True)

def _lookup_pond_area(data, node, params):
    if params['sw_ponding_process'] == 'enabled':
        zone_sw = data['params']['sw_zone_mapping'][node]
        pond_area = data['params']['sw_ponding_area'][zone_sw]
    else:
        pond_area = 0.0
    return pond_area

def _copy_new_bits_into_cat_output(stuff):
    term = "runoff_recharge"
    for cat in stuff.reporting_agg2:
        if "runoff_recharge" in stuff.reporting_agg2[cat]:
            stuff.reporting_agg[cat]["combined_recharge"] += stuff.reporting_agg2[
                cat][term][term]
            stuff.reporting_agg[cat]["combined_str"] -= stuff.reporting_agg2[cat][
                term][term]
            stuff.reporting_agg[cat]["runoff_recharge"] = stuff.reporting_agg2[
                cat][term][term]
