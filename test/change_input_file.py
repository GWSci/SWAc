from swacmod.input_files.input_files_version_1.input_data import load_yaml
import yaml

def change_input_file(input_file, param, change_to):
    file = load_yaml(input_file)
    file[param] = change_to
    with open(input_file, 'w') as f:
        yaml.dump(file, f)
