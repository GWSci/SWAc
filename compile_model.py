import os
import struct
import subprocess as sp
import sys
import datetime
import logging

def compile_model():
    """Compile Cython model."""
    is_compile_required = calculate_is_compile_required()
    code_directory = calculate_code_directory()
    if is_compile_required:
        arch = struct.calcsize('P') * 8
        print('cymodel.pyx modified, recompiling for %d-bit' % arch)
        proc = sp.Popen([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                        cwd=code_directory,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
        if_errors_report_and_exit(proc)

def calculate_code_directory():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, 'swacmod')

def calculate_is_compile_required():
    code_directory = calculate_code_directory()
    mod_c = get_modified_time(os.path.join(code_directory, 'cymodel.c'))
    mod_pyx = get_modified_time(os.path.join(code_directory, 'cymodel.pyx'))
    return mod_pyx >= mod_c

def get_modified_time(path):
    """Get the datetime a file was modified."""
    try:
        mod = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        mod = datetime.datetime(mod.year, mod.month, mod.day, mod.hour,
                                mod.minute, mod.second)
    except OSError:
        logging.warning('Could not find %s, set modified time to 1/1/1901',
                        path)
        mod = datetime.datetime(1901, 1, 1, 0, 0, 0)
    return mod

def if_errors_report_and_exit(proc):
    if proc.returncode != 0:
        print('Could not compile C extensions:')
        print('%s' % proc.stdout.read())
        print('%s' % proc.stderr.read())
        sys.exit(proc.returncode)

if __name__ == "__main__":
    compile_model()
