import multiprocessing
import swacmod.input_files.input_files_version_2.checks as c
from swacmod.input_files.parsed_input_data import ParsedInputData
from functools import partial

def val_num_cores(errors, data, name):
    c.validate_min_exclusive(errors, [data["params"][name]], name, 0)
    c.validate_max_inclusive(errors, [data["params"][name]], name, multiprocessing.cpu_count())

def val_start_date(errors, data, name):
    dat = data["params"][name]
    try:
        dat.strftime('%d/%m/%Y')
    except:
        raise

def val_time_periods(errors, data, name):
    for time_range in data["params"][name]:
        if len(time_range) != 2:
            msg = 'Parameter "%s" requires arrays of length 2'
            errors.append(msg % name)
        is_range_ints = (
            isinstance(time_range[0], int) and isinstance(time_range[1], int))
        if not is_range_ints:
            continue
        if not time_range[0] < time_range[1]:
            msg = 'Parameter "%s" requires start_date < end_date'
            errors.append(msg % name)

def val_output_individual(errors, data, name):
    oin = data["params"][name]
    ids = set(range(1, data["params"]["num_nodes"] + 1))
    if not all(i in ids for i in oin):
        msg = ('Parameter "%s" requires all node ids to be'
               + '1 <= x <= ' "num_nodes")
        errors.append(msg % name)

def validate_min_exclusive(min_exclusive, errors, data, name):
    c.validate_min_exclusive(errors, [data["params"][name]], name, min_exclusive)

def val_spatial_output_date(errors, data, name):
    dat = data["params"][name]
    if dat is None or dat != "mean":
        return

def validate_locs(errors, data, name):
    param = data["params"][name]
    if param != {0: 0}:
        tot = len(data["params"][name]) + 1
        # TODO Issue #163. Bug with key validation for swabs_locs and swdis_locs.
        # c.validate_keys(errors, param, name, data["specs"][name]["type"], range(1, tot))
        c.validate_min_inclusive(errors, param.values(), "zone in %s" % name, 1)
        c.validate_max_inclusive(errors, param.values(), "zone in %s" % name, tot)
        c.validate_min_inclusive(errors, param.keys(), "node in %s" % name, 1)
        c.validate_max_inclusive(errors, param.keys(), "node in %s" % name, data["params"]["num_nodes"])

def val_node_areas(errors, data, name):
    c.validate_keys(errors, data["params"][name], name, data["specs"][name]["type"], range(1, data["params"]["num_nodes"] + 1))
    c.validate_min_inclusive(errors, data["params"][name].values(), name, 0)

def validate_zone_mapping_C(zone_name, errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper(zone_name, 0, "zone in %s", param_values, errors, data, name)

def validate_zone_mapping_A(zone_name, min_inclusive, errors, data, name):
    param_values = [i[0] for i in data["params"][name].values()]
    _validate_zone_mapping_helper(zone_name, min_inclusive, "zone in %s", param_values, errors, data, name)

def validate_zone_mapping_B(zone_name, errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper(zone_name, 0, "%s", param_values, errors, data, name)

def _validate_zone_mapping_helper(zone_name, min_inclusive, message_format, param_values, errors, data, name):
    param = data["params"][name]
    tot = data["params"]["num_nodes"]
    tzn = data["params"][zone_name]
    c.validate_keys(errors, param, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_min_inclusive(errors, param_values, message_format % name, min_inclusive)
    c.validate_max_inclusive(errors, param_values, message_format % name, len(tzn))

# TODO This is never called. Should it be?
def val_single_cell_swrecharge_zone_mapping(errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper("single_cell_swrecharge_zone_names", 0, "%s", param_values, errors, data, name)

def validate_constraints(errors, data, name):
    x = data["params"][name]
    c.validate_constraints(errors, [x], name, data["specs"][name]["constraints"])

def val_free_throughfall(errors, data, name):
    fth = data["params"][name]
    tot = len(data["params"]["canopy_zone_names"])
    c.validate_keys(errors, fth, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_min_inclusive(errors, fth.values(), name, 0)
    c.validate_max_inclusive(errors, fth.values(), name, 1.0)

def val_max_canopy_storage(errors, data, name):
    mcs = data["params"][name]
    tot = len(data["params"]["canopy_zone_names"])
    c.validate_keys(errors, mcs, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_min_inclusive(errors, mcs.values(), name, 0)

def validate_snow(process_name, list_length, errors, data, name):
    if data["params"][process_name] == "disabled":
        return
    snp = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, snp, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, snp, name, data["specs"][name]["type"], [list_length], )
    if type(snp) == dict:
        c.validate_min_inclusive(errors, [i[0] for i in snp.values() if len(i) > 0], "starting_snow_pack in %s" % name, 0)

def val_rapid_runoff_params(errors, data, name):
    rrp = data["params"][name]
    rzn = data["params"]["rapid_runoff_zone_names"]
    c.validate_list_length(errors, 
        rrp, name, data["specs"][name]["type"],
        [len(rzn)]
    )
    keys = ["class_smd", "class_ri", "values"]
    for zone in rrp:
        c.validate_keys(errors, zone, name, [dict], keys)
        if "values" in zone:
            c.validate_list_length(errors, zone["values"], name,[list, list],[len(zone["class_ri"]), len(zone["class_smd"])],)
            c.validate_min_inclusive(errors, [i for j in zone["values"] for i in j], '"values" in "%s"' % name, 0)
            c.validate_max_inclusive(errors, [i for j in zone["values"] for i in j], '"values" in "%s"' % name, 1)

def val_rorecharge_process(errors, data, name):
    rop = data['params'][name]
    rrp = data['params']['rapid_runoff_process']
    if rop == 'enabled' and rrp == 'disabled':
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        errors.append(msg % (name, 'rapid_runoff_process'))

def val_single_cell_swrecharge_proportion(errors, data, name):
    rrp = data['params'][name]
    rzn = data['params']['single_cell_swrecharge_zone_names']
    c.validate_keys(errors, rrp, name, data['specs'][name]['type'], range(1, 13))
    c.validate_list_length(errors, rrp, name, data['specs'][name]['type'],[len(rzn)])
    c.validate_min_inclusive(errors, [j for i in rrp.values() for j in i], name, 0)
    c.validate_max_inclusive(errors, [j for i in rrp.values() for j in i], name, 1.0)

# TODO This is never called. Should it be?
def val_single_cell_swrecharge_limit(errors, data, name):
    validate_months_and_zones('single_cell_swrecharge_zone_names', errors, data, name)

def validate_months_and_zones(zone_name, errors, data, name):
    param = data['params'][name]
    rzn = data['params'][zone_name]
    c.validate_keys(errors, param, name, data['specs'][name]['type'], range(1, 13))
    c.validate_list_length(errors, param, name, data['specs'][name]['type'], [len(rzn)])

# TODO This is never called. Should it be?
def val_single_cell_swrecharge_activation(errors, data, name):
    validate_months_and_zones('single_cell_swrecharge_zone_names', errors, data, name)

def val_swrecharge_process(errors, data, name):
    rop = data["params"][name]
    rrp = data["params"]["rapid_runoff_process"]
    if rop == "enabled" and rrp == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        errors.append(msg % (name, "rapid_runoff_process"))

def val_swrecharge_proportion(errors, data, name):
    rrp = data["params"][name]
    rzn = data["params"]["swrecharge_zone_names"]
    c.validate_keys(errors, rrp, name, data["specs"][name]["type"], range(1, 13))
    c.validate_list_length(errors, rrp, name, data["specs"][name]["type"],[len(rzn)],)
    c.validate_min_inclusive(errors, [j for i in rrp.values() for j in i], name, 0)
    c.validate_max_inclusive(errors, [j for i in rrp.values() for j in i], name, 1.0)

def val_macropore_process(errors, data, name):
    mpp = data["params"][name]
    rrp = data["params"]["rapid_runoff_process"]
    if mpp == "enabled" and rrp == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        errors.append(msg % (name, "rapid_runoff_process"))

def val_macropore_proportion(errors, data, name):
    mpp = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]
    c.validate_keys(errors, mpp, name, data["specs"][name]["type"], range(1, 13))
    c.validate_list_length(errors, mpp, name, data["specs"][name]["type"],[len(mzn)],)
    c.validate_min_inclusive(errors, [j for i in mpp.values() for j in i], name, 0)
    c.validate_max_inclusive(errors, [j for i in mpp.values() for j in i], name, 1.0)

def val_macropore_recharge(errors, data, name):
    mpr = data["params"][name]
    mzn = data["params"]["macropore_zone_names"]
    c.validate_keys(errors, mpr, name, data["specs"][name]["type"], range(1, 13))
    c.validate_list_length(errors, mpr, name, data["specs"][name]["type"],[len(mzn)],)
    c.validate_min_inclusive(errors, [j for i in mpr.values() for j in i], name, 0)
    c.validate_max_inclusive(errors, [j for i in mpr.values() for j in i], name, 1.0)

def val_soil_static_params(errors, data, name):
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return
    ssp = data["params"][name]
    szn = data["params"]["soil_zone_names"]
    c.validate_keys(errors, ssp, name, data["specs"][name]["type"],["FC", "WP", "p"],)
    c.validate_list_length(errors, ssp, name, data["specs"][name]["type"],[len(szn)],)

def val_smd(errors, data, name):
    smd = data["params"][name]
    szn = data["params"]["soil_zone_names"]
    c.validate_keys(errors, smd, name, data["specs"][name]["type"],["starting_SMD"],)
    c.validate_list_length(errors, smd, name, data["specs"][name]["type"],[len(szn)],)

def val_soil_spatial(errors, data, name):
    sos = data["params"][name]
    soz = data["params"]["soil_zone_names"]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, sos, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, sos, name, data["specs"][name]["type"],[len(soz)],)
    if not all(sum(i) == 1.0 for i in sos.values()):
        msg = 'Parameter "%s" requires the sum of its values to be 1.0'
        errors.append(msg % name)

def val_lu_spatial(errors, data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    lus = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, lus, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, lus, name, data["specs"][name]["type"],[len(lzn.values())],)
    if not all(abs(1 - sum(i)) < 1e-5 for i in lus.values()):
        msg = ('Parameter "%s" requires the sum of its values '
               'to be 1.0 within a tolerance of 1e-5')
        errors.append(msg % name)

def val_zr(errors, data, name):
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return
    zrn = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    c.validate_keys(errors, zrn, name, data["specs"][name]["type"], range(1, 13))
    c.validate_list_length(errors, zrn, name, data["specs"][name]["type"],[len(lzn)],)

def validate_fao_keys_and_list_length(errors, data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    param = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    c.validate_keys(errors, param, name, data["specs"][name]["type"], range(1, 13))
    c.validate_list_length(errors, param, name, data["specs"][name]["type"], [len(lzn)])

def val_percolation_rejection(errors, data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    per = data["params"][name]
    lzn = data["params"]["landuse_zone_names"]
    c.validate_keys(errors, per, name, data["specs"][name]["type"],["percolation_rejection"],)
    c.validate_list_length(errors, per, name, data["specs"][name]["type"],[len(lzn)],)
    c.validate_min_inclusive(errors, list(per.values())[0], name, 0.0)

def val_subroot_leakage_fraction(errors, data, name):
    lea = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, lea, name, data["specs"][name]["type"], range(1, tot + 1))

def validate_keys_and_min_values(errors, data, name):
    nda = data["params"][name]
    tot = len(data["params"]["interflow_zone_names"])
    c.validate_keys(errors, nda, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_min_inclusive(errors, nda.values(), name, 0)

def val_recharge_attenuation_params(errors, data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, rpn, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, rpn, name, data["specs"][name]["type"],[3],)
    c.validate_min_inclusive(errors, [i[1] for i in rpn.values() if len(i) > 1], "release_proportion in %s" % name, 0.0)
    c.validate_max_inclusive(errors, [i[1] for i in rpn.values() if len(i) > 1], "release_proportion in %s" % name, 1.0)

def val_sw_zone_mapping(errors, data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        validate_zone_mapping_B("sw_zone_names", errors, data, name)

def validate_ponding_keys_length_range(errors, data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.validate_keys(errors, rrp, name, data["specs"][name]["type"], range(1, 13))
        c.validate_list_length(errors, rrp, name, data["specs"][name]["type"],[len(rzn)],)
        c.validate_min_inclusive(errors, [j for i in rrp.values() for j in i], name, 0)
        c.validate_max_inclusive(errors, [j for i in rrp.values() for j in i], name, 1.0)

def validate_ponding_keys_length(errors, data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        rrp = data["params"][name]
        rzn = data["params"]["sw_zone_names"]
        c.validate_keys(errors, rrp, name, data["specs"][name]["type"], range(1, 13))
        c.validate_list_length(errors, rrp, name, data["specs"][name]["type"],[len(rzn)],)

def val_sw_ponding_area(errors, data, name):
    if data["params"]['sw_ponding_process'] == "enabled":
        num = data["params"][name]
        values = [i for i in num.values()]
        c.validate_max_inclusive(errors, values, name, 1.0)
        c.validate_min_exclusive(errors, values, name, 0)

def val_sw_params(errors, data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, rpn, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, rpn, name, data["specs"][name]["type"],[2],)
    c.validate_min_inclusive(errors, [i[1] for i in rpn.values() if len(i) > 1], "release_proportion in %s" % name, 0.0)
    c.validate_max_inclusive(errors, [i[1] for i in rpn.values() if len(i) > 1], "release_proportion in %s" % name, 1.0)

def val_routing_topology(errors, data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, rpn, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, rpn, name, data["specs"][name]["type"],[10],)

def val_recharge_node_mapping(errors, data, name):
    rpn = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, rpn, name, data["specs"][name]["type"], range(1, tot + 1))
    c.validate_list_length(errors, rpn, name, data["specs"][name]["type"],[1],)

def val_evt_parameters(errors, data, name):
    rpn = data["params"][name]
    c.validate_list_length(errors, rpn, name, data["specs"][name]["type"],[3],)

def validate_2(params, specs):
    data = {
        "params": params,
        "specs": specs
    }
    errors = []
    warnings = []
    _validate_params(errors, specs, data)
    return ParsedInputData(params, errors, warnings)

def _validate_params(errors, specs, data):
    do_validation(errors, data, val_num_cores, "num_cores")
    do_validation(errors, data, partial(validate_min_exclusive, 0), "num_nodes")
    do_validation(errors, data, val_node_areas, "node_areas")
    do_validation(errors, data, val_start_date, "start_date")
    do_validation(errors, data, val_time_periods, "time_periods")
    do_validation(errors, data, val_output_individual, "output_individual")
    do_validation(errors, data, partial(validate_min_exclusive, 0), "nodes_per_line")
    do_validation(errors, data, partial(validate_min_exclusive, 0.0), "output_fac")
    # val_spatial_output_date,
    do_validation(errors, data, partial(validate_zone_mapping_C, "reporting_zone_names"), "reporting_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_A, "rainfall_zone_names", 1), "rainfall_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "rapid_runoff_zone_names"), "rapid_runoff_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_A, "pe_zone_names", 0), "pe_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "temperature_zone_names"), "temperature_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "tmax_c_zone_names"), "tmax_c_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "tmin_c_zone_names"), "tmin_c_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "tmin_c_zone_names"), "windsp_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_A, "subroot_zone_names", 0), "subroot_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "swrecharge_zone_names"), "swrecharge_zone_mapping")
    do_validation(errors, data, val_sw_zone_mapping, "sw_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "macropore_zone_names"), "macropore_zone_mapping")
    do_validation(errors, data, val_recharge_node_mapping, "recharge_node_mapping")
    do_validation(errors, data, validate_constraints, "macropore_activation_option")
    do_validation(errors, data, val_free_throughfall, "free_throughfall")
    do_validation(errors, data, val_max_canopy_storage, "max_canopy_storage")
    do_validation(errors, data, partial(validate_snow, "snow_process_simple", 3), "snow_params_simple")
    do_validation(errors, data, partial(validate_snow, "snow_process_complex", 10), "snow_params_complex")
    do_validation(errors, data, val_rapid_runoff_params, "rapid_runoff_params")
    do_validation(errors, data, val_swrecharge_process, "swrecharge_process")
    do_validation(errors, data, val_swrecharge_proportion, "swrecharge_proportion")
    do_validation(errors, data, partial(validate_months_and_zones, "swrecharge_zone_names"), "swrecharge_limit")
    do_validation(errors, data, val_macropore_process, "macropore_process")
    do_validation(errors, data, val_macropore_proportion, "macropore_proportion")
    do_validation(errors, data, partial(validate_months_and_zones, "macropore_zone_names"), "macropore_limit")
    do_validation(errors, data, partial(validate_months_and_zones, "macropore_zone_names"), "macropore_activation")
    do_validation(errors, data, val_macropore_recharge, "macropore_recharge")
    do_validation(errors, data, val_soil_static_params, "soil_static_params")
    do_validation(errors, data, val_smd, "smd")
    do_validation(errors, data, val_soil_spatial, "soil_spatial")
    do_validation(errors, data, val_lu_spatial, "lu_spatial")
    do_validation(errors, data, val_zr, "zr")
    do_validation(errors, data, validate_fao_keys_and_list_length, "kc")
    do_validation(errors, data, validate_fao_keys_and_list_length, "taw")
    do_validation(errors, data, validate_fao_keys_and_list_length, "raw")
    do_validation(errors, data, val_percolation_rejection, "percolation_rejection")
    do_validation(errors, data, val_subroot_leakage_fraction, "subroot_leakage_fraction")
    do_validation(errors, data, validate_keys_and_min_values, "init_interflow_store")
    do_validation(errors, data, validate_keys_and_min_values, "interflow_store_bypass")
    do_validation(errors, data, validate_keys_and_min_values, "infiltration_limit")
    do_validation(errors, data, validate_keys_and_min_values, "interflow_decay")
    do_validation(errors, data, val_recharge_attenuation_params, "recharge_attenuation_params")
    do_validation(errors, data, validate_ponding_keys_length_range, "sw_downstream")
    do_validation(errors, data, validate_ponding_keys_length, "sw_activation")
    do_validation(errors, data, validate_ponding_keys_length_range, "sw_bed_infiltration")
    do_validation(errors, data, validate_ponding_keys_length_range, "sw_direct_recharge")
    do_validation(errors, data, validate_ponding_keys_length, "sw_pe_to_open_water")
    do_validation(errors, data, val_sw_params, "sw_params")
    do_validation(errors, data, val_sw_ponding_area, "sw_ponding_area")
    do_validation(errors, data, validate_locs, "swdis_locs")
    do_validation(errors, data, validate_locs, "swabs_locs")
    do_validation(errors, data, val_routing_topology, "routing_topology")
    do_validation(errors, data, validate_constraints, "swdis_f")
    do_validation(errors, data, validate_constraints, "swabs_f")
    do_validation(errors, data, val_evt_parameters, "evt_parameters")
    do_validation(errors, data, validate_constraints, "nevtopt")
    do_validation(errors, data, partial(validate_zone_mapping_B, "interflow_zone_names"), "interflow_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_B, "canopy_zone_names"), "canopy_zone_mapping")

def do_validation(errors, data, function, param):
    if not is_param_skipped(data, param):
        function(errors, data, param)

def is_param_skipped(data, param):
    params = data["params"]
    specs = data["specs"]
    is_skipped = (params[param] == None) or is_alt(specs, params, param)
    return is_skipped

def is_alt(specs, params, param):
    value = params[param]

    if not isinstance(value, str):
        return False

    alt_formats = specs[param].get("alt_format", [])
    for alt in alt_formats:
        suffix = f".{alt}"
        if value.endswith(suffix):
            return True
    return False
