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
