import swacmod.input_files.input_files_version_1.input_data as input_data_v1
import swacmod.input_files.input_files_version_2.input_data as input_data_v2

def scrape_run_name(input_file):
    params = input_data_v1.load_yaml(input_file)
    run_name = params["run_name"]
    return run_name

def read_inputs(specs_file, input_file, input_dir):
    version = detect_version(input_file)
    if (version == 1):
        parsed_input_data = input_data_v1.load_and_validate(specs_file, input_file, input_dir)
    elif (version == 2):
        parsed_input_data = input_data_v2.load_and_validate(specs_file, input_file, input_dir)
    else:
        raise Exception(f"Unknown version: '{version}'.")

    return parsed_input_data.data

def detect_version(input_file):
    params = input_data_v1.load_yaml(input_file)
    version = params.get("version", 1)
    return version

def migrate(data):
    result = data
    if (result.get("version", 1) == 1):
        result = migrate_v1_to_v2(result)
    return result

def migrate_v1_to_v2(data):
    result = data
    result["version"] = 2
    return result
