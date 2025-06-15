#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod check functions."""

# Standard Library
import datetime

# Third Party Libraries
import numpy as np

# Internal modules
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.time_series_data as time_series_data

basestring = str # TODO str should be inlined, but only when the surrounding code can be tested.

MAPPING = {
    (int, int): ["an integer", "integers"],
    (float, int, int): ["a number", "numbers"],
    str: ["a string", "strings"],
    dict: ["a dictionary", "dictionaries"],
    (list, np.ndarray): ["a list", "lists"],
    set: ["a set", "sets"],
    basestring: ["a string", "strings"],
    datetime.datetime: ["a datetime", "datetimes"],
    np.ndarray: ["a numpy array", "numpy arrays"],
}

def expand_t_type(t_type):
    """Expand t_type to all allowed types."""
    if t_type == float:
        t_type = (float, int, int)
    elif t_type == int:
        t_type = (int, int)
    elif t_type == list:
        t_type = (list, np.ndarray, time_series_data.TimeSeriesData)
    return t_type

def check_type(param=None, name=None, t_types=None, len_list=None, keys=None):
    if (t_types is None) or len(t_types) == 0:
        return
    types = [i for i in t_types]

    t_type = types.pop(0)
    t_type = expand_t_type(t_type)

    expanded_list = expand_t_type(list)

    new_len = None
    if t_type == expanded_list and len_list:
        new_len = len_list.pop(0)

    if t_type == expanded_list and new_len and len(param) != new_len:
        msg = 'Parameter "%s" has to be a list of length %d, found %d'
        raise u.ValidationError(msg % (name, new_len, len(param)))

    if t_type == dict and (type(param) == dict) and keys:
        set_keys = set(keys)
        param_keys = set(param.keys())
        if not set_keys.issubset(param_keys):
            msg = 'Parameter "%s" is missing the following keys: %s'
            diff = set_keys - param_keys
            raise u.ValidationError(msg % (name, diff))

    if len(types) > 0 and t_type == dict and (type(param) == dict):
        for value in param.values():
            copy_t = [i for i in types]
            copy_l = []
            if len_list:
                copy_l = [i for i in len_list]
            check_type(
              param=value, name=name, t_types=copy_t, len_list=copy_l,
              keys=keys
            )

    elif len(types) > 0 and t_type in [set, expanded_list]:
        for value in param:
            copy_t = [i for i in types]
            copy_l = []
            if len_list:
                copy_l = [i for i in len_list]
            check_type(
              param=value, name=name, t_types=copy_t, len_list=copy_l,
              keys=keys
            )
    else:
        check_type(param=param, name=name, t_types=types, len_list=len_list, keys=keys)

def check_values_limits(
    values,
    name,
    low_l=None,
    high_l=None,
    include_low=False,
    include_high=False,
    constraints=None
):
    """Check the values are all within two limits."""
    if low_l is not None:
        if not include_low and not all(i > low_l for i in values):
            msg = 'Parameter "%s" requires values > %s'
            raise u.ValidationError(msg % (name, low_l))
        elif include_low and not all(i >= low_l for i in values):
            msg = 'Parameter "%s" requires values >= %s'
            raise u.ValidationError(msg % (name, low_l))

    if high_l is not None:
        if not include_high and not all(i < high_l for i in values):
            msg = 'Parameter "%s" requires values < %s'
            raise u.ValidationError(msg % (name, high_l))
        elif include_high and not all(i <= high_l for i in values):
            msg = 'Parameter "%s" requires values <= %s'
            raise u.ValidationError(msg % (name, high_l))

    if constraints is not None:
        if not all(i in constraints for i in values):
            msg = 'Parameter "%s" requires to be one in %s'
            raise u.ValidationError(msg % (name, constraints))

def validate_max_inclusive(values, name, high_l):
    if not all(i <= high_l for i in values):
        msg = 'Parameter "%s" requires values <= %s'
        raise u.ValidationError(msg % (name, high_l))

def validate_min_exclusive(values, name, low_l):
    if not all(i > low_l for i in values):
        msg = 'Parameter "%s" requires values > %s'
        raise u.ValidationError(msg % (name, low_l))

def validate_min_inclusive(values, name, low_l):
    if not all(i >= low_l for i in values):
        msg = 'Parameter "%s" requires values >= %s'
        raise u.ValidationError(msg % (name, low_l))

def validate_constraints(values, name, constraints):
    if not all(i in constraints for i in values):
        msg = 'Parameter "%s" requires to be one in %s'
        raise u.ValidationError(msg % (name, constraints))
