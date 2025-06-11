from __future__ import print_function

import swacmod.input_files.input_files_version_2.specs as specs_module
from swacmod.input_files.parsed_input_data import ParsedInputData
import os
from dataclasses import dataclass
import numpy as np
import swacmod.input_files.input_files_version_2.time_series_data as time_series_data

def validate_keys(specs, params, input_file):
    result = ParsedInputData(params, [], [])
    result.update(validate_required_fields(params, input_file))
    result.update(validate_no_extra_params(specs_module.make_specs_dictionary(specs), params, input_file))
    return result

def validate_filenames(params, input_dir, is_alt_format, filename_exists):
    errors = []
    warnings = []
    for param in params:
        if is_alt_format[param]:
            absolute = os.path.join(input_dir, params[param])
            if not filename_exists(absolute):
                errors.append(f'Error: Unknown file name: "{params[param]}"')
    return ParsedInputData(params, errors, warnings)

def validate_required_fields(params, input_file):
    errors = []
    warnings = []
    required_fields = specs_module.required_field_names(specs_module.make_specs())
    for field in required_fields:
        if (not field in params):
            errors.append(f'Error: The file "{input_file}" is missing the required field "{field}".')
    return ParsedInputData(params, errors, warnings)

def validate_no_extra_params(specs, params, input_file):
    valid_keys = set(specs.keys())
    found_keys = set(params.keys())
    unrecognised_keys = sorted(found_keys.difference(valid_keys))
    errors = []
    warnings = []

    valid_keys_string = ", ".join([f'"{k}"' for k in sorted(valid_keys)])
    for key in unrecognised_keys:
        errors.append(f'Error: Found the key "{key}" in the file "{input_file}". This key is not valid. A full list of valid keys is: [{valid_keys_string}].')

    return ParsedInputData(params, errors, warnings)

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

@dataclass
class Validation_Config:
    spec: specs_module.Input_Parameter
    params: dict
    errors: list
    key: str
    value: object
    result: ParsedInputData

def validate_matches_spec(specs_object, params, key):
    config = make_validation_config(specs_object, params, key)

    if not validate_spec_present_for_key(config):
        return config.result

    validate_required_key(config)

    if key not in config.params:
        return config.result

    validate_type(config)
    validate_set_member_types(config)
    validate_list_member_types(config)
    validate_list_inner_types(config)
    validate_dictionary_key_types(config)
    validate_dictionary_value_types(config)
    validate_dictionary_value_inner_types(config)
    return config.result

def make_validation_config(specs_object, params, key):
    spec = _find_spec_or_none(specs_object, key)
    errors = []
    value = params.get(key, None)
    result = ParsedInputData(params, errors, [])
    config = Validation_Config(spec, params, errors, key, value, result)
    return config

def validate_spec_present_for_key(config):
    if config.spec == None:
        config.errors.append(f"Error: No spec was found for the key '{config.key}'. Please contact support.")
        return False
    return True

def validate_required_key(config):
    if (config.spec.required) and (config.key not in config.params):
        config.errors.append(f"Error: The required key '{config.key}' could not be found.")

def validate_type(config):
    acceptable_types = find_type_synonyms(config.spec.type[0])
    if type(config.value) not in acceptable_types:
        type_string = _convert_type_to_string(config.spec.type)
        config.errors.append(
            f"Error: The field '{config.key}' must be {type_string}. "
            + f"The value '{config.value}' is invalid.")

def find_type_synonyms(t):
    type_synonyms = {
        int: [int, np.int64],
        float: [float, int, np.float64, np.int64],
        list: [list, np.ndarray, time_series_data.TimeSeriesData],
    }
    acceptable_types = type_synonyms.get(t, [t])
    return acceptable_types

def validate_set_member_types(config):
    iterate_members = lambda: config.value
    expected_member_type = lambda: config.spec.type[1]
    _validate_member_types(
        config, set, iterate_members, "member", expected_member_type)

def validate_list_member_types(config):
    iterate_members = lambda: config.value
    expected_member_type = lambda: config.spec.type[1]
    _validate_member_types(
        config, list, iterate_members, "member", expected_member_type)

def _validate_member_types(
        config,
        collection_type,
        iterate_members,
        member_description,
        expected_member_type):

    is_collection_type = ((type(config.value) == collection_type) 
        and (config.spec.type[0] == collection_type))
    if not is_collection_type:
        return

    acceptable_types = find_type_synonyms(expected_member_type())
    for m in iterate_members():
        if type(m) not in acceptable_types:
            type_string = _convert_type_to_string(config.spec.type)
            config.errors.append(
                f"Error: The field '{config.key}' must be {type_string}. "
                + f"The {member_description} '{m}' is invalid.")
            return

def validate_dictionary_key_types(config):
    iterate_members = lambda: config.value.keys()
    expected_member_type = lambda: int
    _validate_member_types(
        config, dict, iterate_members, "dictionary key", expected_member_type)

def validate_dictionary_value_types(config):
    iterate_members = lambda: config.value.values()
    expected_member_type = lambda: config.spec.type[1]
    _validate_member_types(
        config, dict, iterate_members, "dictionary value", expected_member_type)

def validate_dictionary_value_inner_types(config):
    if len(config.spec.type) < 3:
        return

    iterate_members = lambda: config.value.values()
    expected_member_type = lambda: config.spec.type[2]
    member_description = "inner list value"

    is_collection_type = ((type(config.value) == dict) 
        and (config.spec.type[0] == dict))
    if not is_collection_type:
        return

    acceptable_types = find_type_synonyms(expected_member_type())
    for m in iterate_members():
        if type(m) not in [list, np.ndarray]:
            continue
        for x in m:
            if type(x) not in acceptable_types:
                type_string = _convert_type_to_string(config.spec.type)
                config.errors.append(
                    f"Error: The field '{config.key}' must be {type_string}. "
                    + f"The {member_description} '{x}' is invalid.")
                return

def validate_list_inner_types(config):
    if len(config.spec.type) < 3:
        return

    iterate_members = lambda: config.value
    expected_member_type = lambda: config.spec.type[2]
    member_description = "inner list value"

    is_collection_type = ((type(config.value) == list) 
        and (config.spec.type[0] == list))
    if not is_collection_type:
        return

    acceptable_types = find_type_synonyms(expected_member_type())
    for m in iterate_members():
        if type(m) not in [list, np.ndarray]:
            continue
        for x in m:
            if type(x) not in acceptable_types:
                type_string = _convert_type_to_string(config.spec.type)
                config.errors.append(
                    f"Error: The field '{config.key}' must be {type_string}. "
                    + f"The {member_description} '{x}' is invalid.")
                return

def _convert_type_to_string(spec_type):
    word = convert_type_to_english(spec_type[0])

    article = find_article(word)

    if spec_type[0] == set:
        suffix = f" of {convert_type_to_english(spec_type[1])}s"
    elif spec_type[0] == dict and len(spec_type) == 2:
        suffix = f" mapping integers to {convert_type_to_english(spec_type[1])}s"
    elif spec_type[0] == dict and len(spec_type) == 3:
        suffix = (f" mapping integers"
            + f" to {convert_type_to_english(spec_type[1])}" 
            + f" of {convert_type_to_english(spec_type[2])}s")
    elif spec_type[0] == list and len(spec_type) == 3:
        suffix = (f" of {convert_type_to_english(spec_type[1])}" 
            + f" of {convert_type_to_english(spec_type[2])}s")
    else:
        suffix = ""

    return f"{article} {word}{suffix}"

def convert_type_to_english(t):
    type_to_english = {
        str: "string",
        int: "integer",
        bool: "boolean",
        dict: "dictionary"
    }
    return type_to_english.get(t, t.__name__)

def find_article(word):
    if word[0] in ["a", "e", "i", "o", "u"]:
        return "an"
    return "a"

def _find_spec_or_none(specs_object, key):
    for spec in specs_object:
        if key == spec.name:
            return spec
    return None
