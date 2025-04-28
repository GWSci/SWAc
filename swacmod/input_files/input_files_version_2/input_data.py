# -*- coding: utf-8 -*-
from __future__ import print_function
"""SWAcMod input/output functions."""

# Standard Library
import os
import ast
import sys
import logging
from dataclasses import dataclass

# Third Party Libraries
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from tqdm import tqdm

# Internal modules
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.checks as c
import swacmod.input_files.input_files_version_2.validation as v
import swacmod.input_files.input_files_version_2.finalization as f
import swacmod.input_files.input_files_version_2.time_series_data as time_series_data
import swacmod.input_files.input_files_version_2.specs as specs_module
import swacmod.csv_resource as csv_resource

if sys.version_info > (3,):
    long = int
    raw_input = input

@dataclass
class Validation_Result:
    errors: list
    warnings: list

def load_and_validate(specs_file, input_file, input_dir):
    """Load, finalize and validate model parameters and time series."""
    data = fast_load(specs_file, input_file)
    f.finalize_required_params(data)
    c.check_required(data)
    data = load_params_from_yaml(specs_file, input_file, input_dir, tqdm)

    f.finalize_params(data)
    f.finalize_series(data)
    c.check_required(data)
    v.validate_params(data)
    v.validate_series(data)

    return data

def fast_load(specs_file, input_file):
    "Load the main input file, without loading the files the parameters point to"
    specs = specs_module.make_specs_dictionary(specs_module.make_specs())
    params = load_yaml(input_file)
    for key in specs:
        if key not in params:
            params[key] = None
    params, series = _get_series_from_params(params)
    data = {"specs": specs, "series": series, "params": params}
    return data

def _default_file_open(filename):
    return open(filename, "r")

def load_yaml(filein, file_opener=_default_file_open):
    """Load a YAML file, lowercase its keys."""
    logging.debug("\t\tLoading %s", filein)

    with file_opener(filein) as fp:
        yml = yaml.load(fp, Loader=Loader)
    try:
        keys = list(yml.keys())
    except AttributeError:
        return yml

    for key in keys:
        if isinstance(key, str):
            if not key.islower():
                new_key = key.lower()
                value = yml.pop(key)
                yml[new_key] = value
    return yml

def _get_series_from_params(params):
    """Get the params dictionary and separate it into series and params"""
    series = {}
    keys = [i for i in params if i.endswith("_ts")]
    for key in keys:
        series[key] = params.pop(key)
    return params, series

def load_params_from_yaml(specs_file, input_file, input_dir, tqdm, file_opener=_default_file_open):
    """Load model specifications, parameters and time series."""
    logging.info("\tLoading parameters and time series")

    specs = load_yaml(specs_file, file_opener)
    params = load_yaml(input_file, file_opener)

    no_list = ([
        "node_areas",
        "free_throughfall",
        "max_canopy_storage",
        "subsoilzone_leakage_fraction",
    ] + [i for i in params if "zone_names" in i] + [
        i for i in params if ("zone_mapping" in i or "_locs" in i) and i not in
        ["rainfall_zone_mapping", "pe_zone_mapping", "subroot_zone_mapping"]
    ])

    validate_no_extra_params(specs, params, input_file)

    _load_alt_formats(input_dir, tqdm, file_opener, specs, params, no_list)
    _supply_null_values_for_missing_fields(specs, params)


    params, series = _get_series_from_params(params)

    data = {"specs": specs, "series": series, "params": params}

    return data

def validate(params, input_file):
    errors = []
    warnings = []
    required_fields = specs_module.required_field_names(specs_module.make_specs())
    for field in required_fields:
        if (not field in params):
            errors.append(f'Error: The file "{input_file}" is missing the required field "{field}".')
    return Validation_Result(errors, warnings)

def validate_no_extra_params(specs, params, input_file):
    valid_keys = set(specs.keys())
    found_keys = set(params.keys())
    unrecognised_keys = sorted(found_keys.difference(valid_keys))
    errors = []

    valid_keys_string = ", ".join([f'"{k}"' for k in sorted(valid_keys)])
    for key in unrecognised_keys:
        errors.append(f'Error: Found the key "{key}" in the file "{input_file}". This key is not valid. A full list of valid keys is: [{valid_keys_string}].')

    if len(errors) > 0:
        message = "\n".join(errors)
        raise Exception(message)

def _load_alt_formats(input_dir, tqdm, file_opener, specs, params, no_list):
    for param in tqdm(params, desc="SWAcMod load params     "):
        if isinstance(params[param], str) and "alt_format" in specs[param]:
            absolute = os.path.join(input_dir, params[param])
            ext = params[param].split(".")[-1]
            if ext not in specs[param]["alt_format"] and ext != "numpydumpy":
                continue
            if _use_time_series_data(param):
                load_temp_file_backed_array(params, param, absolute, ext)
            elif ext == "csv":
                load_csv_alt_format(params, no_list, param, absolute)
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

def load_temp_file_backed_array(params, param, absolute, ext):
    base_path = params["temp_file_backed_array_directory"]
    params[param] = time_series_data.load_time_series_data(base_path, param, absolute, ext)

def load_csv_alt_format(params, no_list, param, absolute):
    try:
        with csv_resource.reader_for(absolute) as reader:
            rows = [[ast.literal_eval(j) for j in row]
                                for row in reader]
    except Exception as err:
        msg = "Could not import %s: %s" % (param, err)
        raise u.InputOutputError(msg)
    try:
        if _use_array_directly(param):
            params[param] = rows
        else:
            if param not in no_list:
                params[param] = dict(
                                (row[0], row[1:]) for row in rows)
            else:
                params[param] = dict(
                                (row[0], row[1]) for row in rows)
    except Exception as err:
        msg = "Could not import %s: %s" % (param, err)
        raise u.InputOutputError(msg)

def _use_array_directly(param):
    return (param.endswith("_ts")
            or param == "time_periods"
            or param == "historical_time_periods"
            or param == "historical_mi_array_kg_per_time_period")

def _supply_null_values_for_missing_fields(specs, params):
    for key in specs:
        if key not in params:
            params[key] = None

def load_yml_alt_format(params, param, absolute, file_opener):
    parsed_yaml = load_yaml_file_contents(param, absolute, file_opener)
    validate_alt_yml(param, absolute, parsed_yaml)
    params[param] = parsed_yaml[param]

def load_yaml_file_contents(param, absolute, file_opener):
    try:
        parsed_yaml = load_yaml(absolute, file_opener)
    except Exception as err:
        msg = "Could not import %s: %s" % (param, err)
        raise u.InputOutputError(msg)
    return parsed_yaml

def validate_alt_yml(param, absolute, parsed_yaml):
    errors = []

    keys = extract_keys_or_empty_list(parsed_yaml)

    if (not param in keys):
        errors.append(f'Error: Could not find the key "{param}" in the file "{absolute}".')

    for key in keys:
        if (key != param):
            errors.append(f'Error: Found the key "{key}" in the file "{absolute}". The only key should be "{param}". Common causes of this error are typos and forgetting to indent the key-value pairs in a map.')

    if len(errors) > 0:
        message = "\n".join(errors)
        raise Exception(message)

def extract_keys_or_empty_list(something_that_might_have_keys):
    try:
        return something_that_might_have_keys.keys()
    except Exception:
        return []
