import subprocess
import ast

version_filename = '_version.py'
commit_id_filename = '_commit_id.py'
build_time_filename = '_build_time.py'

def _default_file_open(filename, how='r'):
        return open(filename, how)

def increment_version():
    old_version = get_old_version()
    new_version = get_new_version(old_version)
    write_new_version(new_version)
    subprocess.run(['git', 'add', version_filename])

def get_old_version(filename=version_filename, file_open=_default_file_open):
    with file_open(filename, 'r') as file:
        old_version = file.readlines()[0]
    old_version = old_version.replace('\n', '')
    old_version = old_version.replace(' ', '')
    old_version = old_version.replace('version', '')
    old_version = old_version.replace('=', '')
    old_version = ast.literal_eval(old_version)
    return old_version

def get_new_version(old_version):
    new_version = old_version.copy()
    new_version[-1] += 1
    return new_version

def write_new_version(new_version, filename=version_filename, file_open=_default_file_open):
    with file_open(filename, 'w') as file:
        file.write(f'version = {new_version}')

if (__name__ == "__main__"):
    increment_version()
