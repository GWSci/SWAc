import unittest
from swacmod.input_output import load_params_from_yaml
import tempfile
import shutil
import swacmod.utils as u
import os

class Test_Files_Failing_Validation(unittest.TestCase):
    def test_load_and_validate_a_model_with_a_jumbled_up_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copytree(u.CONSTANTS['TEST_INPUT_DIR'], temp_dir, dirs_exist_ok=True)
            with open(os.path.join(temp_dir, 'time_periods.csv'), 'r') as file:
                lines = file.readlines()
            lines[0] += '043.das'
            with open(os.path.join(temp_dir, 'time_periods.csv'), 'w') as file:
                file.writelines(lines)
            try:
                load_params_from_yaml(input_file=os.path.join(temp_dir, 'input.yml'),
                                    input_dir=temp_dir)
            except Exception as err:
                self.assertIn('time_periods', str(err))
    
    


