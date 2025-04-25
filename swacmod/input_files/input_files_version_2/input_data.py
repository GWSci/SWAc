# -*- coding: utf-8 -*-
from __future__ import print_function
"""SWAcMod input/output functions."""

# Standard Library
import os
import ast
import sys
import logging

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
import swacmod.csv_resource as csv_resource

if sys.version_info > (3,):
    long = int
    raw_input = input

def load_yaml(filein):
    """Load a YAML file, lowercase its keys."""
    logging.debug("\t\tLoading %s", filein)

    with open(filein, "r") as fp:
        yml = yaml.load(fp, Loader=Loader)
    try:
        keys = yml.keys()
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

def load_params_from_yaml(specs_file, input_file, input_dir, tqdm):
    """Load model specifications, parameters and time series."""
    logging.info("\tLoading parameters and time series")

    specs = load_yaml(specs_file)
    params = load_yaml(input_file)

    no_list = ([
        "node_areas",
        "free_throughfall",
        "max_canopy_storage",
        "subsoilzone_leakage_fraction",
    ] + [i for i in params if "zone_names" in i] + [
        i for i in params if ("zone_mapping" in i or "_locs" in i) and i not in
        ["rainfall_zone_mapping", "pe_zone_mapping", "subroot_zone_mapping"]
    ])

    for param in tqdm(params, desc="SWAcMod load params     "):
        if isinstance(params[param], str) and "alt_format" in specs[param]:
            absolute = os.path.join(input_dir, params[param])
            ext = params[param].split(".")[-1]
            if ext not in specs[param]["alt_format"] and ext != "numpydumpy":
                continue
            if _use_time_series_data(param):
                base_path = params["temp_file_backed_array_directory"]
                params[param] = time_series_data.load_time_series_data(base_path, param, absolute, ext)
            elif ext == "csv":
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
            elif ext == "yml":
                load_yml_alt_format(params, param, absolute)
    for key in specs:
        if key not in params:
            params[key] = None

    params, series = _get_series_from_params(params)

    data = {"specs": specs, "series": series, "params": params}

    return data

def load_yml_alt_format(params, param, absolute):
    try:
        params[param] = load_yaml(absolute)[param]
    except Exception as err:
        msg = "Could not import %s: %s" % (param, err)
        raise u.InputOutputError(msg)

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

def _use_array_directly(param):
    return (param.endswith("_ts")
            or param == "time_periods"
            or param == "historical_time_periods"
            or param == "historical_mi_array_kg_per_time_period")

def fast_load(specs_file, input_file):
    "Load the main input file, without loading the files the parameters point to"
    specs = load_yaml(specs_file)
    params = load_yaml(input_file)
    for key in specs:
        if key not in params:
            params[key] = None
    params, series = _get_series_from_params(params)
    data = {"specs": specs, "series": series, "params": params}
    return data

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
