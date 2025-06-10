from __future__ import print_function

import swacmod.input_files.input_files_version_2.specs as specs_module
from swacmod.input_files.parsed_input_data import ParsedInputData
import os
from datetime import date, datetime

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

def validate_matches_spec(specs_object, params, key):
    errors = []
    result = ParsedInputData(params, errors, [])
    spec = _find_spec_or_none(specs_object, key)

    if spec == None:
        errors.append(f"Error: No spec was found for the key '{key}'. Please contact support.")
        return result

    if (spec.required) and (key not in params):
        errors.append(f"Error: The required key '{key}' could not be found.")

    if key not in params:
        return result

    value = params[key]

    type_synonyms = {
        float: [float, int]
    }
    acceptable_types = type_synonyms.get(spec.type[0], [spec.type[0]])
    if type(value) not in acceptable_types:
        type_string = _convert_type_to_string(spec.type)
        errors.append(f"Error: The field '{key}' must be {type_string}. The value '{value}' is invalid.")
    return result

def _convert_type_to_string(spec_type):
    type_to_english = {
        str: "string",
        int: "integer",
        bool: "boolean",
        float: "float",
        date: "date",
    }
    word = type_to_english.get(spec_type[0], f"{type(spec_type[0])}")
    word = f"{type(spec_type[0])}"
    if spec_type[0] == str:
        word = "string"
    if spec_type[0] == int:
        word = "integer"
    if spec_type[0] == bool:
        word = "boolean"
    if spec_type[0] == float:
        word = "float"
    if spec_type[0] == date:
        word = "date"
    if spec_type[0] == datetime:
        word = "datetime"

    article = find_article(word)

    return f"{article} {word}"

def find_article(word):
    if word[0] in ["a", "e", "i", "o", "u"]:
        return "an"
    return "a"

def _find_spec_or_none(specs_object, key):
    for spec in specs_object:
        if key == spec.name:
            return spec
    return None
