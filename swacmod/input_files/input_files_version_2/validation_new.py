import multiprocessing
import swacmod.input_files.input_files_version_2.checks as c
from swacmod.input_files.parsed_input_data import ParsedInputData
from functools import partial

def validate_min_exclusive(min_exclusive, errors, data, name):
    c.validate_min_exclusive(errors, [data["params"][name]], name, min_exclusive)

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

def validate_zone_mapping_c(zone_name, errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper(zone_name, 0, "zone in %s", param_values, errors, data, name)

def validate_zone_mapping_a(zone_name, min_inclusive, errors, data, name):
    param_values = [i[0] for i in data["params"][name].values()]
    _validate_zone_mapping_helper(zone_name, min_inclusive, "zone in %s", param_values, errors, data, name)

def validate_zone_mapping_b(zone_name, errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper(zone_name, 0, "%s", param_values, errors, data, name)

def _validate_zone_mapping_helper(zone_name, min_inclusive, message_format, param_values, errors, data, name):
    zone_names = data["params"][zone_name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_min_inclusive(errors, param_values, message_format % name, min_inclusive)
    c.validate_max_inclusive(errors, param_values, message_format % name, len(zone_names))

def validate_constraints(errors, data, name):
    param = data["params"][name]
    c.validate_constraints(errors, [param], name, data["specs"][name]["constraints"])

def validate_snow(process_name, list_length, errors, data, name):
    if data["params"][process_name] == "disabled":
        return
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"], [list_length], )
    if type(param) == dict:
        c.validate_min_inclusive(errors, [i[0] for i in param.values() if len(i) > 0], "starting_snow_pack in %s" % name, 0)

def validate_mutually_exclusive_with_rapid_runoff_process(errors, data, name):
    param = data["params"][name]
    other_process = data["params"]["rapid_runoff_process"]
    if param == "enabled" and other_process == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        errors.append(msg % (name, "rapid_runoff_process"))

def validate_months_and_zones(zone_name_key, errors, data, name):
    param = data["params"][name]
    c.validate_keys(errors, param, name, data["specs"][name]["type"], range(1, 13))
    validate_list_length_equals_zone_name_count(zone_name_key, errors, data, name)

def validate_keys_length_min_max(zone_name_key, errors, data, name):
    param = data["params"][name]
    validate_months_and_zones(zone_name_key, errors, data, name)
    c.validate_min_inclusive(errors, [j for i in param.values() for j in i], name, 0)
    c.validate_max_inclusive(errors, [j for i in param.values() for j in i], name, 1.0)

def validate_fao_keys_and_list_length(errors, data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    validate_months_and_zones("landuse_zone_names", errors, data, name)

def validate_keys_are_nodes(errors, data, name):
    param = data["params"][name]
    tot = data["params"]["num_nodes"]
    c.validate_keys(errors, param, name, data["specs"][name]["type"], range(1, tot + 1))

def validate_keys_and_min_values(errors, data, name):
    param = data["params"][name]
    validate_keys_are_zone_name_numbers("interflow_zone_names", errors, data, name)
    c.validate_min_inclusive(errors, param.values(), name, 0)

def validate_release_proportion_min_and_max(errors, data, name):
    param = data["params"][name]
    c.validate_min_inclusive(errors, [i[1] for i in param.values() if len(i) > 1], "release_proportion in %s" % name, 0.0)
    c.validate_max_inclusive(errors, [i[1] for i in param.values() if len(i) > 1], "release_proportion in %s" % name, 1.0)

def validate_ponding_keys_length_range(errors, data, name):
    if data["params"]["sw_ponding_process"] == "enabled":
        validate_keys_length_min_max("sw_zone_names", errors, data, name)

def validate_ponding_keys_length(errors, data, name):
    if data["params"]["sw_ponding_process"] == "enabled":
        validate_months_and_zones("sw_zone_names", errors, data, name)

def validate_list_length_equals_zone_name_count(zone_name_key, errors, data, name):
    param = data["params"][name]
    zone_names = data["params"][zone_name_key]
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[len(zone_names)],)

def validate_keys_are_zone_name_numbers(zone_name_key, errors, data, name):
    param = data["params"][name]
    tot = len(data["params"][zone_name_key])
    c.validate_keys(errors, param, name, data["specs"][name]["type"], range(1, tot + 1))

def val_num_cores(errors, data, name):
    validate_min_exclusive(0, errors, data, name)
    c.validate_max_inclusive(errors, [data["params"][name]], name, multiprocessing.cpu_count())

def val_start_date(errors, data, name):
    param = data["params"][name]
    try:
        param.strftime("%d/%m/%Y")
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
    param = data["params"][name]
    ids = set(range(1, data["params"]["num_nodes"] + 1))
    if not all(i in ids for i in param):
        msg = ('Parameter "%s" requires all node ids to be'
               + '1 <= x <= ' "num_nodes")
        errors.append(msg % name)

def val_spatial_output_date(errors, data, name):
    dat = data["params"][name]
    if dat is None or dat != "mean":
        return

def val_node_areas(errors, data, name):
    validate_keys_are_nodes(errors, data, name)
    c.validate_min_inclusive(errors, data["params"][name].values(), name, 0)

def val_free_throughfall(errors, data, name):
    param = data["params"][name]
    validate_keys_are_zone_name_numbers("canopy_zone_names", errors, data, name)
    c.validate_min_inclusive(errors, param.values(), name, 0)
    c.validate_max_inclusive(errors, param.values(), name, 1.0)

def val_max_canopy_storage(errors, data, name):
    param = data["params"][name]
    validate_keys_are_zone_name_numbers("canopy_zone_names", errors, data, name)
    c.validate_min_inclusive(errors, param.values(), name, 0)

def val_rapid_runoff_params(errors, data, name):
    param = data["params"][name]
    validate_list_length_equals_zone_name_count("rapid_runoff_zone_names", errors, data, name)
    keys = ["class_smd", "class_ri", "values"]
    for zone in param:
        c.validate_keys(errors, zone, name, [dict], keys)
        if "values" in zone:
            c.validate_list_length(errors, zone["values"], name,[list, list],[len(zone["class_ri"]), len(zone["class_smd"])],)
            c.validate_min_inclusive(errors, [i for j in zone["values"] for i in j], '"values" in "%s"' % name, 0)
            c.validate_max_inclusive(errors, [i for j in zone["values"] for i in j], '"values" in "%s"' % name, 1)

def val_single_cell_swrecharge_proportion(errors, data, name):
    zone_name_key = "single_cell_swrecharge_zone_names"
    validate_keys_length_min_max(zone_name_key, errors, data, name)

def val_soil_static_params(errors, data, name):
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return
    param = data["params"][name]
    c.validate_keys(errors, param, name, data["specs"][name]["type"],["FC", "WP", "p"],)
    validate_list_length_equals_zone_name_count("soil_zone_names", errors, data, name)

def val_smd(errors, data, name):
    param = data["params"][name]
    c.validate_keys(errors, param, name, data["specs"][name]["type"],["starting_SMD"],)
    validate_list_length_equals_zone_name_count("soil_zone_names", errors, data, name)

def val_soil_spatial(errors, data, name):
    param = data["params"][name]
    soz = data["params"]["soil_zone_names"]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[len(soz)],)
    if not all(sum(i) == 1.0 for i in param.values()):
        msg = 'Parameter "%s" requires the sum of its values to be 1.0'
        errors.append(msg % name)

def val_lu_spatial(errors, data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    validate_list_length_equals_zone_name_count("landuse_zone_names", errors, data, name)
    if not all(abs(1 - sum(i)) < 1e-5 for i in param.values()):
        msg = ('Parameter "%s" requires the sum of its values '
               'to be 1.0 within a tolerance of 1e-5')
        errors.append(msg % name)

def val_zr(errors, data, name):
    if (
        data["params"]["fao_process"] == "disabled"
        or data["params"]["fao_input"] == "l"
    ):
        return
    validate_months_and_zones("landuse_zone_names", errors, data, name)

def val_percolation_rejection(errors, data, name):
    if data["params"]["fao_process"] == "disabled":
        return
    param = data["params"][name]
    c.validate_keys(errors, param, name, data["specs"][name]["type"],["percolation_rejection"],)
    validate_list_length_equals_zone_name_count("landuse_zone_names", errors, data, name)
    c.validate_min_inclusive(errors, list(param.values())[0], name, 0.0)

def val_recharge_attenuation_params(errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[3],)
    validate_release_proportion_min_and_max(errors, data, name)

def val_sw_zone_mapping(errors, data, name):
    if data["params"]["sw_ponding_process"] == "enabled":
        validate_zone_mapping_b("sw_zone_names", errors, data, name)

def val_sw_ponding_area(errors, data, name):
    if data["params"]["sw_ponding_process"] == "enabled":
        num = data["params"][name]
        values = [i for i in num.values()]
        c.validate_max_inclusive(errors, values, name, 1.0)
        c.validate_min_exclusive(errors, values, name, 0)

def val_sw_params(errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[2],)
    validate_release_proportion_min_and_max(errors, data, name)

def val_routing_topology(errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[10],)

def val_recharge_node_mapping(errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[1],)

def val_evt_parameters(errors, data, name):
    param = data["params"][name]
    c.validate_list_length(errors, param, name, data["specs"][name]["type"],[3],)

# TODO This is never called. Should it be?
def val_single_cell_swrecharge_zone_mapping(errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper("single_cell_swrecharge_zone_names", 0, "%s", param_values, errors, data, name)

# TODO This is never called. Should it be?
def val_single_cell_swrecharge_limit(errors, data, name):
    validate_months_and_zones("single_cell_swrecharge_zone_names", errors, data, name)

# TODO This is never called. Should it be?
def val_single_cell_swrecharge_activation(errors, data, name):
    validate_months_and_zones("single_cell_swrecharge_zone_names", errors, data, name)

def validate_2(params, specs):
    data = {
        "params": params,
        "specs": specs
    }
    errors = []
    warnings = []
    _validate_params(errors, data)
    return ParsedInputData(params, errors, warnings)

def _validate_params(errors, data):
    do_validation(errors, data, val_num_cores, "num_cores")
    do_validation(errors, data, partial(validate_min_exclusive, 0), "num_nodes")
    do_validation(errors, data, val_node_areas, "node_areas")
    do_validation(errors, data, val_start_date, "start_date")
    do_validation(errors, data, val_time_periods, "time_periods")
    do_validation(errors, data, val_output_individual, "output_individual")
    do_validation(errors, data, partial(validate_min_exclusive, 0), "nodes_per_line")
    do_validation(errors, data, partial(validate_min_exclusive, 0.0), "output_fac")
    # val_spatial_output_date,
    do_validation(errors, data, partial(validate_zone_mapping_c, "reporting_zone_names"), "reporting_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_a, "rainfall_zone_names", 1), "rainfall_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "rapid_runoff_zone_names"), "rapid_runoff_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_a, "pe_zone_names", 0), "pe_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "temperature_zone_names"), "temperature_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "tmax_c_zone_names"), "tmax_c_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "tmin_c_zone_names"), "tmin_c_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "tmin_c_zone_names"), "windsp_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_a, "subroot_zone_names", 0), "subroot_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "swrecharge_zone_names"), "swrecharge_zone_mapping")
    do_validation(errors, data, val_sw_zone_mapping, "sw_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "macropore_zone_names"), "macropore_zone_mapping")
    do_validation(errors, data, val_recharge_node_mapping, "recharge_node_mapping")
    do_validation(errors, data, validate_constraints, "macropore_activation_option")
    do_validation(errors, data, val_free_throughfall, "free_throughfall")
    do_validation(errors, data, val_max_canopy_storage, "max_canopy_storage")
    do_validation(errors, data, partial(validate_snow, "snow_process_simple", 3), "snow_params_simple")
    do_validation(errors, data, partial(validate_snow, "snow_process_complex", 10), "snow_params_complex")
    do_validation(errors, data, val_rapid_runoff_params, "rapid_runoff_params")
    do_validation(errors, data, validate_mutually_exclusive_with_rapid_runoff_process, "swrecharge_process")
    do_validation(errors, data, partial(validate_keys_length_min_max, "swrecharge_zone_names"), "swrecharge_proportion")
    do_validation(errors, data, partial(validate_months_and_zones, "swrecharge_zone_names"), "swrecharge_limit")
    do_validation(errors, data, validate_mutually_exclusive_with_rapid_runoff_process, "macropore_process")
    do_validation(errors, data, partial(validate_keys_length_min_max, "macropore_zone_names"), "macropore_proportion")
    do_validation(errors, data, partial(validate_months_and_zones, "macropore_zone_names"), "macropore_limit")
    do_validation(errors, data, partial(validate_months_and_zones, "macropore_zone_names"), "macropore_activation")
    do_validation(errors, data, partial(validate_keys_length_min_max, "macropore_zone_names"), "macropore_recharge")
    do_validation(errors, data, val_soil_static_params, "soil_static_params")
    do_validation(errors, data, val_smd, "smd")
    do_validation(errors, data, val_soil_spatial, "soil_spatial")
    do_validation(errors, data, val_lu_spatial, "lu_spatial")
    do_validation(errors, data, val_zr, "zr")
    do_validation(errors, data, validate_fao_keys_and_list_length, "kc")
    do_validation(errors, data, validate_fao_keys_and_list_length, "taw")
    do_validation(errors, data, validate_fao_keys_and_list_length, "raw")
    do_validation(errors, data, val_percolation_rejection, "percolation_rejection")
    do_validation(errors, data, validate_keys_are_nodes, "subroot_leakage_fraction")
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
    do_validation(errors, data, partial(validate_zone_mapping_b, "interflow_zone_names"), "interflow_zone_mapping")
    do_validation(errors, data, partial(validate_zone_mapping_b, "canopy_zone_names"), "canopy_zone_mapping")

def do_validation(errors, data, function, param):
    if not is_param_skipped(data, param):
        function(errors, data, param)

def is_param_skipped(data, param):
    params = data["params"]
    specs = data["specs"]
    is_skipped = (params[param] is None) or is_alt(specs, params, param)
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
