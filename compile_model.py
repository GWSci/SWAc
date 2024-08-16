import os
import struct
import subprocess as sp
import sys
import datetime
import logging
from swacmod import utils as u

def compile_model():
    """Compile Cython model."""
    mod_c = get_modified_time(os.path.join(u.CONSTANTS['CODE_DIR'], 'cymodel.c'))
    mod_pyx = get_modified_time(
        os.path.join(u.CONSTANTS['CODE_DIR'], 'cymodel.pyx'))
    if mod_pyx >= mod_c:
        arch = struct.calcsize('P') * 8
        print('cymodel.pyx modified, recompiling for %d-bit' % arch)
        proc = sp.Popen([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                        cwd=u.CONSTANTS['CODE_DIR'],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
        if proc.returncode != 0:
            print('Could not compile C extensions:')
            print('%s' % proc.stdout.read())
            print('%s' % proc.stderr.read())
            sys.exit(proc.returncode)

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

if __name__ == "__main__":
    compile_model()
