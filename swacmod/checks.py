#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod check functions."""

# Standard Library
import os
import datetime

# Internal modules
from . import utils as u

MAPPING = {(int, long): ['an integer', 'integers'],
           (float, int, long): ['a number', 'numbers'],
           str: ['a string', 'strings'],
           basestring: ['a string', 'strings'],
           dict: ['a dictionary', 'dictionaries'],
           list: ['a list', 'lists'],
           set: ['a set', 'sets'],
           basestring: ['a string', 'strings'],
           datetime.datetime: ['a datetime', 'datetimes']}


###############################################################################
def check_type(param=None, name=None, t_types=None, len_list=None, keys=None):
    """Check the parameter is of type t_type."""
    while t_types:

        t_type = t_types.pop(0)
        if t_type == float:
            t_type = (float, int, long)
        elif t_type == int:
            t_type = (int, long)

        new_len = None
        if t_type == list and len_list:
            new_len = len_list.pop(0)

        if not isinstance(param, t_type):
            msg = 'Parameter "%s" has to be %s, found a %s instead'
            raise u.ValidationError(msg % (name, MAPPING[t_type][0],
                                           type(param)))

        if t_type == list and new_len and len(param) != new_len:
            msg = 'Parameter "%s" has to be a list of length %d, found %d'
            raise u.ValidationError(msg % (name, new_len, len(param)))

        if t_type == dict and keys and set(keys) != set(param.keys()):
            msg = 'Parameter "%s" is missing the following ids: %s'
            diff = set(keys) - set(param.keys())
            raise u.ValidationError(msg % (name, diff))

        if len(t_types) > 0 and t_type == dict:
            for value in param.values():
                copy_t = [i for i in t_types]
                copy_l = []
                if len_list:
                    copy_l = [i for i in len_list]
                check_type(param=value, name=name, t_types=copy_t,
                           len_list=copy_l, keys=keys)
            t_types = []
            len_list = []

        elif len(t_types) > 0 and t_type == list:
            for value in param:
                copy_t = [i for i in t_types]
                copy_l = []
                if len_list:
                    copy_l = [i for i in len_list]
                check_type(param=value, name=name, t_types=copy_t,
                           len_list=copy_l, keys=keys)
            t_types = []
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
        key = ('series' if param.endswith('ts') else 'params')
        if param not in data[key] or not data[key][param]:
            if data['specs'][param]['required']:
                raise u.ValidationError('Parameter "%s" is required' % param)
            else:
                u.normalize_default_value(data, param)
