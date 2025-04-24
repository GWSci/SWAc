from swacmod.input_files.input_files_version_1 import input_data as input_data_v1
from swacmod.input_files.input_files_version_2 import input_data as input_data_v2

def read_inputs(specs_file, input_file, input_dir):
    version = detect_version(input_file)
    if (version == 1):
        return input_data_v1.load_and_validate(specs_file, input_file, input_dir)
    elif (version == 2):
        return input_data_v2.load_and_validate(specs_file, input_file, input_dir)
    else:
        raise Exception(f"Unknown version: '{version}'.")

def detect_version(input_file):
    params = input_data_v1.load_yaml(input_file)
    version = params.get("version", 1)
    return version
