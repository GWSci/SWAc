import os
import struct
import subprocess as sp
import sys
from swacmod import utils as u

def compile_model():
    """Compile Cython model."""
    mod_c = u.get_modified_time(os.path.join(u.CONSTANTS['CODE_DIR'], 'cymodel.c'))
    mod_pyx = u.get_modified_time(
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
        boo = True
    else:
        boo = False
    return boo

if __name__ == "__main__":
    compile_model()
