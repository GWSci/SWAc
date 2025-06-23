import multiprocessing
import swacmod.input_files.input_files_version_2.checks as c
from swacmod.input_files.parsed_input_data import ParsedInputData
from functools import partial
import swacmod.input_files.input_files_version_2.validation_context as validation_context

def validate_min_exclusive(min_exclusive, errors, data, name):
    c.validate_min_exclusive(errors, [data["params"][name]], name, min_exclusive)

def validate_param_values_min_inclusive(min_inclusive, errors, data, name, name_in_message):
    param = data["params"][name]
    c.validate_min_inclusive(errors, param.values(), name_in_message, min_inclusive)

def validate_param_values_min_exclusive(min_inclusive, errors, data, name, name_in_message):
    param = data["params"][name]
    c.validate_min_exclusive(errors, param.values(), name_in_message, min_inclusive)

def validate_param_values_max_inclusive(max_inclusive, errors, data, name, name_in_message):
    param = data["params"][name]
    c.validate_max_inclusive(errors, param.values(), name_in_message, max_inclusive)

def validate_locs(errors, data, name):
    param = data["params"][name]
    if param != {0: 0}:
        tot = len(data["params"][name]) + 1
        # TODO Issue #163. Bug with key validation for swabs_locs and swdis_locs.
        # _validate_keys(range(1, tot), errors, data, name)
        validate_param_values_min_inclusive(1, errors, data, name, "zone in %s" % name)
        validate_param_values_max_inclusive(tot, errors, data, name, "zone in %s" % name)
        c.validate_min_inclusive(errors, param.keys(), "node in %s" % name, 1)
        c.validate_max_inclusive(errors, param.keys(), "node in %s" % name, data["params"]["num_nodes"])

def validate_zone_mapping_a(zone_name, min_inclusive, errors, data, name):
    param_values = [i[0] for i in data["params"][name].values()]
    _validate_zone_mapping_helper(zone_name, min_inclusive, param_values, errors, data, name)

def validate_zone_mapping_b(zone_name, errors, data, name):
    param_values = data["params"][name].values()
    _validate_zone_mapping_helper(zone_name, 0, param_values, errors, data, name)

def _validate_zone_mapping_helper(zone_name, min_inclusive, param_values, errors, data, name):
    zone_names = data["params"][zone_name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_min_inclusive(errors, param_values, "zone in %s" % name, min_inclusive)
    c.validate_max_inclusive(errors, param_values, "zone in %s" % name, len(zone_names))

def validate_constraints(errors, data, name):
    param = data["params"][name]
    c.validate_constraints(errors, [param], name, data["specs"][name]["constraints"])

def validate_snow(list_length, errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    c.validate_list_length(errors, param, name, data["specs"][name]["type"], [list_length], )
    if type(param) == dict:
        c.validate_min_inclusive(errors, [i[0] for i in param.values() if len(i) > 0], "starting_snow_pack in %s" % name, 0)

def validate_cannot_be_enabled_without_rapid_runoff_process(errors, data, name):
    param = data["params"][name]
    other_process = data["params"]["rapid_runoff_process"]
    if param == "enabled" and other_process == "disabled":
        msg = 'Cannot set "%s" to "enabled" and "%s" to "disabled"'
        errors.append(msg % (name, "rapid_runoff_process"))

def validate_months_and_zones(zone_name_key, errors, data, name):
    _validate_keys(range(1, 13), errors, data, name)
    validate_list_length_equals_zone_name_count(zone_name_key, errors, data, name)

def validate_keys_length_min_max(zone_name_key, errors, data, name):
    param = data["params"][name]
    validate_months_and_zones(zone_name_key, errors, data, name)
    c.validate_min_inclusive(errors, [j for i in param.values() for j in i], name, 0)
    c.validate_max_inclusive(errors, [j for i in param.values() for j in i], name, 1.0)

def validate_keys_are_nodes(errors, data, name):
    tot = data["params"]["num_nodes"]
    _validate_keys(range(1, tot + 1), errors, data, name)

def validate_keys_and_min_values(errors, data, name):
    validate_keys_are_zone_name_numbers("interflow_zone_names", errors, data, name)
    validate_param_values_min_inclusive(0, errors, data, name, name)

def validate_release_proportion_min_and_max(errors, data, name):
    param = data["params"][name]
    c.validate_min_inclusive(errors, [i[1] for i in param.values() if len(i) > 1], "release_proportion in %s" % name, 0.0)
    c.validate_max_inclusive(errors, [i[1] for i in param.values() if len(i) > 1], "release_proportion in %s" % name, 1.0)

def validate_list_length_equals_zone_name_count(zone_name_key, errors, data, name):
    zone_names = data["params"][zone_name_key]
    validate_list_length([len(zone_names)], errors, data, name)

def validate_keys_are_zone_name_numbers(zone_name_key, errors, data, name):
    tot = len(data["params"][zone_name_key])
    _validate_keys(range(1, tot + 1), errors, data, name)

def validate_list_length(expected_list_lengths, errors, data, name):
    param = data["params"][name]
    c.validate_list_length(errors, param, name, data["specs"][name]["type"] ,expected_list_lengths)

def _validate_keys(expected_keys, errors, data, name):
    param = data["params"][name]
    c.validate_keys(errors, param, name, data["specs"][name]["type"], expected_keys)

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
    validate_param_values_min_inclusive(0, errors, data, name, name)

def val_free_throughfall(errors, data, name):
    validate_keys_are_zone_name_numbers("canopy_zone_names", errors, data, name)
    validate_param_values_min_inclusive(0, errors, data, name, name)
    validate_param_values_max_inclusive(1.0, errors, data, name, name)

def val_max_canopy_storage(errors, data, name):
    validate_keys_are_zone_name_numbers("canopy_zone_names", errors, data, name)
    validate_param_values_min_inclusive(0, errors, data, name, name)

def val_rapid_runoff_params(errors, data, name):
    param = data["params"][name]
    validate_list_length_equals_zone_name_count("rapid_runoff_zone_names", errors, data, name)
    keys = ["class_smd", "class_ri", "values"]
    for zone in param:
        _validate_keys(keys, errors, data, name)
        if "values" in zone:
            c.validate_list_length(errors, zone["values"], name,[list, list],[len(zone["class_ri"]), len(zone["class_smd"])],)
            c.validate_min_inclusive(errors, [i for j in zone["values"] for i in j], '"values" in "%s"' % name, 0)
            c.validate_max_inclusive(errors, [i for j in zone["values"] for i in j], '"values" in "%s"' % name, 1)

def val_single_cell_swrecharge_proportion(errors, data, name):
    zone_name_key = "single_cell_swrecharge_zone_names"
    validate_keys_length_min_max(zone_name_key, errors, data, name)

def val_soil_static_params(errors, data, name):
    if data["params"]["fao_input"] == "l":
        return
    _validate_keys(["FC", "WP", "p"], errors, data, name)
    validate_list_length_equals_zone_name_count("soil_zone_names", errors, data, name)

def val_smd(errors, data, name):
    _validate_keys(["starting_SMD"], errors, data, name)
    validate_list_length_equals_zone_name_count("soil_zone_names", errors, data, name)

def val_soil_spatial(errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    validate_list_length_equals_zone_name_count("soil_zone_names", errors, data, name)
    if not all(sum(i) == 1.0 for i in param.values()):
        msg = 'Parameter "%s" requires the sum of its values to be 1.0'
        errors.append(msg % name)

def val_lu_spatial(errors, data, name):
    param = data["params"][name]
    validate_keys_are_nodes(errors, data, name)
    validate_list_length_equals_zone_name_count("landuse_zone_names", errors, data, name)
    if not all(abs(1 - sum(i)) < 1e-5 for i in param.values()):
        msg = ('Parameter "%s" requires the sum of its values '
               'to be 1.0 within a tolerance of 1e-5')
        errors.append(msg % name)

def val_zr(errors, data, name):
    if data["params"]["fao_input"] == "l":
        return
    validate_months_and_zones("landuse_zone_names", errors, data, name)

def val_percolation_rejection(errors, data, name):
    param = data["params"][name]
    _validate_keys(["percolation_rejection"], errors, data, name)
    validate_list_length_equals_zone_name_count("landuse_zone_names", errors, data, name)
    c.validate_min_inclusive(errors, list(param.values())[0], name, 0.0)

def val_recharge_attenuation_params(errors, data, name):
    validate_keys_are_nodes(errors, data, name)
    validate_list_length([3], errors, data, name)
    validate_release_proportion_min_and_max(errors, data, name)

def val_sw_ponding_area(errors, data, name):
    validate_param_values_max_inclusive(1.0, errors, data, name, name)
    validate_param_values_min_exclusive(0, errors, data, name, name) # TODO Is this a bug? It seems like it would be sensible for zero to be allowed.

def val_sw_params(errors, data, name):
    validate_keys_are_nodes(errors, data, name)
    validate_list_length([2], errors, data, name)
    validate_release_proportion_min_and_max(errors, data, name)

def val_routing_topology(errors, data, name):
    validate_keys_are_nodes(errors, data, name)
    validate_list_length([10], errors, data, name)

def val_recharge_node_mapping(errors, data, name):
    validate_keys_are_nodes(errors, data, name)
    validate_list_length([1], errors, data, name)

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
    factory = validation_context.Validation_Context_Factory(errors, data)
    all_validation_contexts = [

        factory.make(val_num_cores, "num_cores"),
        factory.make(partial(validate_min_exclusive, 0), "num_nodes"),
        factory.make(val_node_areas, "node_areas"),
        factory.make(val_start_date, "start_date"),
        factory.make(val_time_periods, "time_periods"),
        factory.make(val_output_individual, "output_individual"),
        factory.make(partial(validate_min_exclusive, 0), "nodes_per_line"),
        factory.make(partial(validate_min_exclusive, 0.0), "output_fac"),
        # val_spatial_output_date,
        factory.make(partial(validate_zone_mapping_b, "reporting_zone_names"), "reporting_zone_mapping"),
        factory.make(partial(validate_zone_mapping_a, "rainfall_zone_names", 1), "rainfall_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "rapid_runoff_zone_names"), "rapid_runoff_zone_mapping"),
        factory.make(partial(validate_zone_mapping_a, "pe_zone_names", 0), "pe_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "temperature_zone_names"), "temperature_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "tmax_c_zone_names"), "tmax_c_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "tmin_c_zone_names"), "tmin_c_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "tmin_c_zone_names"), "windsp_zone_mapping"),
        factory.make(partial(validate_zone_mapping_a, "subroot_zone_names", 0), "subroot_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "swrecharge_zone_names"), "swrecharge_zone_mapping"),
        factory.make_with_process_guard(partial(validate_zone_mapping_b, "sw_zone_names"), "sw_zone_mapping", "sw_ponding_process"),
        factory.make(partial(validate_zone_mapping_b, "macropore_zone_names"), "macropore_zone_mapping"),
        factory.make(val_recharge_node_mapping, "recharge_node_mapping"),
        factory.make(validate_constraints, "macropore_activation_option"),
        factory.make(val_free_throughfall, "free_throughfall"),
        factory.make(val_max_canopy_storage, "max_canopy_storage"),
        factory.make_with_process_guard(partial(validate_snow, 3), "snow_params_simple", "snow_process_simple"),
        factory.make_with_process_guard(partial(validate_snow, 10), "snow_params_complex", "snow_process_complex"),
        factory.make(val_rapid_runoff_params, "rapid_runoff_params"),
        factory.make(validate_cannot_be_enabled_without_rapid_runoff_process, "swrecharge_process"),
        factory.make(partial(validate_keys_length_min_max, "swrecharge_zone_names"), "swrecharge_proportion"),
        factory.make(partial(validate_months_and_zones, "swrecharge_zone_names"), "swrecharge_limit"),
        factory.make(validate_cannot_be_enabled_without_rapid_runoff_process, "macropore_process"),
        factory.make(partial(validate_keys_length_min_max, "macropore_zone_names"), "macropore_proportion"),
        factory.make(partial(validate_months_and_zones, "macropore_zone_names"), "macropore_limit"),
        factory.make(partial(validate_months_and_zones, "macropore_zone_names"), "macropore_activation"),
        factory.make(partial(validate_keys_length_min_max, "macropore_zone_names"), "macropore_recharge"),
        factory.make_with_process_guard(val_soil_static_params, "soil_static_params", "fao_process"),
        factory.make(val_smd, "smd"),
        factory.make(val_soil_spatial, "soil_spatial"),
        factory.make_with_process_guard(val_lu_spatial, "lu_spatial", "fao_process"),
        factory.make_with_process_guard(val_zr, "zr", "fao_process"),
        factory.make_with_process_guard(partial(validate_months_and_zones, "landuse_zone_names"), "kc", "fao_process"),
        factory.make_with_process_guard(partial(validate_months_and_zones, "landuse_zone_names"), "taw", "fao_process"),
        factory.make_with_process_guard(partial(validate_months_and_zones, "landuse_zone_names"), "raw", "fao_process"),
        factory.make_with_process_guard(val_percolation_rejection, "percolation_rejection", "fao_process"),
        factory.make(validate_keys_are_nodes, "subroot_leakage_fraction"),
        factory.make(validate_keys_and_min_values, "init_interflow_store"),
        factory.make(validate_keys_and_min_values, "interflow_store_bypass"),
        factory.make(validate_keys_and_min_values, "infiltration_limit"),
        factory.make(validate_keys_and_min_values, "interflow_decay"),
        factory.make(val_recharge_attenuation_params, "recharge_attenuation_params"),
        factory.make_with_process_guard(partial(validate_keys_length_min_max, "sw_zone_names"), "sw_downstream", "sw_ponding_process"),
        factory.make_with_process_guard(partial(validate_months_and_zones, "sw_zone_names"), "sw_activation", "sw_ponding_process"),
        factory.make_with_process_guard(partial(validate_keys_length_min_max, "sw_zone_names"), "sw_bed_infiltration", "sw_ponding_process"),
        factory.make_with_process_guard(partial(validate_keys_length_min_max, "sw_zone_names"), "sw_direct_recharge", "sw_ponding_process"),
        factory.make_with_process_guard(partial(validate_months_and_zones, "sw_zone_names"), "sw_pe_to_open_water", "sw_ponding_process"),
        factory.make(val_sw_params, "sw_params"),
        factory.make_with_process_guard(val_sw_ponding_area, "sw_ponding_area", "sw_ponding_process"),
        factory.make(validate_locs, "swdis_locs"),
        factory.make(validate_locs, "swabs_locs"),
        factory.make(val_routing_topology, "routing_topology"),
        factory.make(validate_constraints, "swdis_f"),
        factory.make(validate_constraints, "swabs_f"),
        factory.make(partial(validate_list_length, [3]), "evt_parameters"),
        factory.make(validate_constraints, "nevtopt"),
        factory.make(partial(validate_zone_mapping_b, "interflow_zone_names"), "interflow_zone_mapping"),
        factory.make(partial(validate_zone_mapping_b, "canopy_zone_names"), "canopy_zone_mapping"),
    ]

    for vc in all_validation_contexts:
        vc.do_validation()
