import unittest
import tempfile
import swacmod.utils as u
import os
import shutil
from swacmod.input_output import load_and_validate
from test.change_input_file import change_input_file

specs_file = u.CONSTANTS['SPECS_FILE']

class Test_pe_ts_Finalisation(unittest.TestCase):
    def test_canopy_and_fao_are_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copytree(u.CONSTANTS['TEST_INPUT_DIR'], temp_dir, dirs_exist_ok=True)
            test_input_file = os.path.join(temp_dir, 'input.yml')
            change_input_file(test_input_file, 'fao_process:', 'fao_process: disabled\n')
            change_input_file(test_input_file, 'canopy_process:', 'canopy_process: disabled\n')
            try:
                data = load_and_validate(specs_file=specs_file,
                                        input_file=test_input_file,
                                        input_dir=temp_dir)
                actual = 0
            except:
                actual = 1
        self.assertEqual(0, actual)
    
    def test_canopy_is_disabled_and_fao_is_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copytree(u.CONSTANTS['TEST_INPUT_DIR'], temp_dir, dirs_exist_ok=True)
            test_input_file = os.path.join(temp_dir, 'input.yml')
            change_input_file(test_input_file, 'fao_process:', 'fao_process: enabled\n')
            change_input_file(test_input_file, 'canopy_process:', 'canopy_process: disabled\n')
            try:
                data = load_and_validate(specs_file=specs_file,
                                        input_file=test_input_file,
                                        input_dir=temp_dir)
                actual = 0
            except:
                actual = 1
        self.assertEqual(0, actual)