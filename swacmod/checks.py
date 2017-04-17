#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod check functions."""

# Standard Library
import os
import datetime

# Third Party Libraries
import numpy as np

# Internal modules
from . import utils as u

MAPPING = {(int, long): ['an integer', 'integers'],
           (float, int, long): ['a number', 'numbers'],
           str: ['a string', 'strings'],
           basestring: ['a string', 'strings'],
           dict: ['a dictionary', 'dictionaries'],
           (list, np.ndarray): ['a list', 'lists'],
           set: ['a set', 'sets'],
           basestring: ['a string', 'strings'],
           datetime.datetime: ['a datetime', 'datetimes'],
           np.ndarray: ['a numpy array', 'numpy arrays']}


###############################################################################
def expand_t_type(t_type):
    """Expand t_type to all allowed types."""
    if t_type == float:
        t_type = (float, int, long)
    elif t_type == int:
        t_type = (int, long)
    elif t_type == list:
        t_type = (list, np.ndarray)

    return t_type


###############################################################################
def check_type(param=None, name=None, t_types=None, len_list=None, keys=None):
    """Check the parameter is of type t_type."""
    types = [i for i in t_types]
    while types:

        t_type = types.pop(0)
        t_type = expand_t_type(t_type)

        new_len = None
        if t_type == (list, np.ndarray) and len_list:
            new_len = len_list.pop(0)

        if not isinstance(param, t_type):
            msg = 'Parameter "%s" has to be %s, found a %s instead'
            raise u.ValidationError(msg % (name, MAPPING[t_type][0],
                                           type(param)))

        if t_type == (list, np.ndarray) and new_len and len(param) != new_len:
            msg = 'Parameter "%s" has to be a list of length %d, found %d'
            raise u.ValidationError(msg % (name, new_len, len(param)))

        if t_type == dict and keys:
            set_keys = set(keys)
            param_keys = set(param.keys())
            if set_keys != param_keys and len(set_keys) > len(param_keys):
                msg = 'Parameter "%s" is missing the following keys: %s'
                diff = set_keys - param_keys
                raise u.ValidationError(msg % (name, diff))

        if len(types) > 0 and t_type == dict:
            for value in param.values():
                copy_t = [i for i in types]
                copy_l = []
                if len_list:
                    copy_l = [i for i in len_list]
                check_type(param=value, name=name, t_types=copy_t,
                           len_list=copy_l, keys=keys)
            types = []
            len_list = []

        elif len(types) > 0 and t_type in [set, (list, np.ndarray)]:
            for value in param:
                copy_t = [i for i in types]
                copy_l = []
                if len_list:
                    copy_l = [i for i in len_list]
                check_type(param=value, name=name, t_types=copy_t,
                           len_list=copy_l, keys=keys)
            types = []
            len_list = []


###############################################################################
def check_values_limits(values=None, name=None, low_l=None, high_l=None,
                        include_low=False, include_high=False,
                        constraints=None):
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


###############################################################################
def check_path(path):
    """Convert path to absolute if it isn't."""
    new_path = path
    if not os.path.isabs(path):
        new_path = os.path.join(u.CONSTANTS['INPUT_DIR'], path)
    return new_path


###############################################################################
def check_required(data):
    """Check that all required parameters have been provided."""
    for param in data['specs']:
        if not data['specs'][param]['required']:
            continue
        key = ('series' if param.endswith('ts') else 'params')
        if data[key][param] is None:
            raise u.ValidationError('Parameter "%s" is required' % param)
