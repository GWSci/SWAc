from __future__ import print_function

import os
import sys
import logging

from tqdm import tqdm

import swacmod.utils as u
import swacmod.input_files.input_files_version_2.loader as loader
import swacmod.input_files.input_files_version_2.validator as validator
import swacmod.input_files.input_files_version_2.validation as v
import swacmod.input_files.input_files_version_2.validation_new as validation_new
import swacmod.input_files.input_files_version_2.finalization as f
import swacmod.input_files.input_files_version_2.specs as specs_module
from swacmod.input_files.parsed_input_data import ParsedInputData

if sys.version_info > (3,):
    long = int
    raw_input = input

def _default_file_open(filename):
    return open(filename, "r")

def load_and_validate(input_file, input_dir, file_opener=_default_file_open):
    """Load, finalize and validate model parameters and time series."""
    logging.info("\tLoading parameters and time series")
    specs_object = specs_module.make_specs()
    specs = specs_module.make_specs_dictionary(specs_object)
    params = loader.load_yaml(input_file, file_opener)
    validation_result = ParsedInputData(params, [], [])

    validation_result.update(validator.validate_keys(specs_object, params, input_file))
    if validation_result.has_errors():
        return ParsedInputData(params, validation_result.errors, validation_result.warnings)

    _supply_null_values_for_missing_fields(specs, params)
    validation_result.update(validation_new.validate_2(params, specs))
    if validation_result.has_errors():
        return validation_result

    data = convert_params_and_specs_into_data(specs, params)
    f.finalize_required_params(data)

    load_alt_formats(input_dir, file_opener, specs, params)

    data = convert_params_and_specs_into_data(specs, params)

    f.finalize_params(data)
    f.finalize_series(data)
    v.validate_params(data)
    v.validate_series(data)

    return validation_result.update(ParsedInputData(data, [], []))

def convert_params_and_specs_into_data(specs, params):
    specs_copy = dict(specs)
    params_copy = dict(params)
    params_copy, series = _get_series_from_params(params_copy)
    data = {"specs": specs_copy, "series": series, "params": params_copy}
    return data

def load_alt_formats(input_dir, file_opener, specs, params):
    no_list = _make_no_list(params)
    _load_alt_formats(input_dir, tqdm, file_opener, specs, params, no_list)

def _supply_null_values_for_missing_fields(specs, params):
    for key in specs:
        if key not in params:
            params[key] = None

def _get_series_from_params(params):
    """Get the params dictionary and separate it into series and params"""
    series = {}
    keys = [i for i in params if i.endswith("_ts")]
    for key in keys:
        series[key] = params.pop(key)
    return params, series

def _make_no_list(params):
    no_list = ([
        "node_areas",
        "free_throughfall",
        "max_canopy_storage",
        "subsoilzone_leakage_fraction",
    ] + [i for i in params if "zone_names" in i] + [
        i for i in params if ("zone_mapping" in i or "_locs" in i) and i not in
        ["rainfall_zone_mapping", "pe_zone_mapping", "subroot_zone_mapping"]
    ])
    
    return no_list

def _load_alt_formats(input_dir, tqdm, file_opener, specs, params, no_list):
    for param in tqdm(params, desc="SWAcMod load params     "):
        should_load_alt_format = False
        might_be_alt_format = isinstance(params[param], str) and "alt_format" in specs[param]
        if might_be_alt_format:
            absolute = os.path.join(input_dir, params[param])
            ext = params[param].split(".")[-1]
            is_not_alt_format = ext not in specs[param]["alt_format"] and ext != "numpydumpy"
            is_alt_format = not is_not_alt_format
            if is_alt_format:
                should_load_alt_format = True
        
        if should_load_alt_format:
            load_alt_format(file_opener, params, no_list, param, absolute, ext)

def load_alt_format(file_opener, params, no_list, param, absolute, ext):
    if _use_time_series_data(param):
        loader.load_temp_file_backed_array(params, param, absolute, ext)
    elif ext == "csv":
        loader.load_csv_alt_format(params, no_list, param, absolute)
    elif ext == "yml":
        load_yml_alt_format(params, param, absolute, file_opener)

def _use_time_series_data(param):
    return param in [
        "historical_mi_array_kg_per_time_period",
        "interflow_decay_ts",
        "percolation_rejection_ts",
        "rainfall_ts",
        "temperature_ts",
        "tmax_c_ts",
        "tmin_c_ts",
        "pe_ts",
        "windsp_ts",
    ]

def load_yml_alt_format(params, param, absolute, file_opener):
    parsed_yaml = loader.load_yaml_file_contents(param, absolute, file_opener)
    validator.validate_alt_yml(param, absolute, parsed_yaml)
    params[param] = parsed_yaml[param]
