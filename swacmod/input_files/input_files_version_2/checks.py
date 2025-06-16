import numpy as np
import swacmod.utils as u
import swacmod.input_files.input_files_version_2.time_series_data as time_series_data

def validate_keys(errors, param, name, t_types, keys):
    if (t_types is None) or len(t_types) == 0:
        return

    t_type = _expand_t_type(t_types[0])
    expanded_list = _expand_t_type(list)

    if t_type == dict and (type(param) == dict) and keys:
        _validate_keys(param, name, keys)

    if t_type == dict and (type(param) == dict):
        for value in param.values():
            validate_keys(errors, value, name, t_types[1:], keys)
    elif t_type in [expanded_list]:
        for value in param:
            validate_keys(errors, value, name, t_types[1:], keys)

def _expand_t_type(t_type):
    if t_type == float:
        t_type = (float, int, int)
    elif t_type == int:
        t_type = (int, int)
    elif t_type == list:
        t_type = (list, np.ndarray, time_series_data.TimeSeriesData)
    return t_type

def _validate_keys(param, name, keys):
    set_keys = set(keys)
    param_keys = set(param.keys())
    if not set_keys.issubset(param_keys):
        msg = 'Parameter "%s" is missing the following keys: %s'
        diff = set_keys - param_keys
        raise u.ValidationError(msg % (name, diff))

def validate_list_length(errors, param, name, t_types, len_list):
    if (t_types is None) or len(t_types) == 0:
        return

    t_type = _expand_t_type(t_types[0])
    expanded_list = _expand_t_type(list)

    if t_type == expanded_list and type(param) in expanded_list and len_list:
        _validate_list_length(param, name, len_list[0])

    if t_type == dict and (type(param) == dict):
        for value in param.values():
            validate_list_length(errors, value, name, t_types[1:], len_list)
    elif t_type in [expanded_list]:
        for value in param:
            validate_list_length(errors, value, name, t_types[1:], _tail(len_list))

def _validate_list_length(param, name, length):
    if length and (len(param) != length):
        msg = 'Parameter "%s" has to be a list of length %d, found %d'
        raise u.ValidationError(msg % (name, length, len(param)))

def _tail(a_list):
    if a_list and len(a_list) > 0:
        return a_list[1:]
    else:
        return a_list

def validate_max_inclusive(errors, values, name, high_l):
    if not all(i <= high_l for i in values):
        msg = 'Parameter "%s" requires values <= %s'
        raise u.ValidationError(msg % (name, high_l))

def validate_min_exclusive(errors, values, name, low_l):
    if not all(i > low_l for i in values):
        msg = 'Parameter "%s" requires values > %s'
        raise u.ValidationError(msg % (name, low_l))

def validate_min_inclusive(errors, values, name, low_l):
    if not all(i >= low_l for i in values):
        msg = 'Parameter "%s" requires values >= %s'
        raise u.ValidationError(msg % (name, low_l))

def validate_constraints(errors, values, name, constraints):
    if not all(i in constraints for i in values):
        msg = 'Parameter "%s" requires to be one in %s'
        raise u.ValidationError(msg % (name, constraints))
