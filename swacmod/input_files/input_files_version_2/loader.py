from __future__ import print_function
"""SWAcMod input/output functions."""

# Standard Library
import ast
import logging
from dataclasses import dataclass

# Third Party Libraries
import yaml

# Internal modules
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.validation as v
import swacmod.input_files.input_files_version_2.finalization as f
import swacmod.input_files.input_files_version_2.time_series_data as time_series_data
import swacmod.csv_resource as csv_resource

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from tqdm import tqdm

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

def load_yaml_file_contents(param, absolute, file_opener):
    try:
        parsed_yaml = load_yaml(absolute, file_opener)
    except Exception as err:
        msg = "Could not import %s: %s" % (param, err)
        raise u.InputOutputError(msg)
    return parsed_yaml
