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

# Internal modules
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.validation as v
import swacmod.input_files.input_files_version_2.validation_new as validation_new
import swacmod.input_files.input_files_version_2.finalization as f
import swacmod.input_files.input_files_version_2.time_series_data as time_series_data
import swacmod.input_files.input_files_version_2.specs as specs_module
from swacmod.input_files.parsed_input_data import ParsedInputData
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
