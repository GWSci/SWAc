import unittest
import tempfile
import swacmod.utils as u
import os
import shutil
from swacmod.input_data import load_and_validate
from test.change_input_file import change_input_file

class Test_pe_ts_Finalisation(unittest.TestCase):
    def test_canopy_and_fao_are_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pairs = [['fao_process', 'disabled'], ['canopy_process', 'disabled']]
            test_input_file = create_and_change_test_input_file(temp_dir, pairs)
            actual = load_test_input_file(temp_dir, test_input_file)
        self.assertEqual(0, actual)

    def test_canopy_is_disabled_and_fao_is_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pairs = [['fao_process', 'enabled'], ['canopy_process', 'disabled']]
            test_input_file = create_and_change_test_input_file(temp_dir, pairs)
            actual = load_test_input_file(temp_dir, test_input_file)
        self.assertEqual(0, actual)
    
    def test_canopy_is_enabled_and_fao_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pairs = [['fao_process', 'disabled'], ['canopy_process', 'enabled']]
            test_input_file = create_and_change_test_input_file(temp_dir, pairs)
            actual = load_test_input_file(temp_dir, test_input_file)
        self.assertEqual(0, actual)

def create_and_change_test_input_file(temp_dir, pairs):
    test_input_file = os.path.join(temp_dir, 'input.yml')
    shutil.copytree(u.CONSTANTS['TEST_INPUT_DIR'], temp_dir, dirs_exist_ok=True)
    for p in pairs: change_input_file(test_input_file, p[0], p[1])
    return test_input_file

def load_test_input_file(temp_dir, test_input_file):
    try:
        data = load_and_validate(specs_file=u.CONSTANTS['SPECS_FILE'],
                                input_file=test_input_file,
                                input_dir=temp_dir)
        return 0
    except:
        return 1