import unittest
import io
from contextlib import contextmanager
import csv
from swacmod.input_files.input_files_version_2.input_data import _make_no_list
import swacmod.input_files.input_files_version_2.loader as loader

class Test_CSVs_Are_Not_Read_As_lists(unittest.TestCase):
    def test_init_interflow_store_csv_input_returns_floats(self):
        param = 'init_interflow_store'
        params = get_params(param)
        for val in params[param].values():
            self.assertIsInstance(val, float)

    def test_interflow_store_bypass_csv_input_returns_floats(self):
        param = 'interflow_store_bypass'
        params = get_params(param)
        for val in params[param].values():
            self.assertIsInstance(val, float)
    
    def test_infiltration_limit_csv_input_returns_floats(self):
        param = 'infiltration_limit'
        params = get_params(param)
        for val in params[param].values():
            self.assertIsInstance(val, float)
    
    def test_interflow_decay_csv_input_returns_floats(self):
        param = 'interflow_decay'
        params = get_params(param)
        for val in params[param].values():
            self.assertIsInstance(val, float)

def get_params(param):
    filename = "potato.csv"
    file_contents = "1,1.0\n2,2.0\n3,3.0"
    params = {param: filename}
    no_list = _make_no_list(params)
    file_opener = make_mock_csv_loader({filename: file_contents})
    loader.load_csv_alt_format(params, no_list, param, filename, file_opener)
    return params

def make_mock_csv_loader(file_contents):
    @contextmanager
    def mock_loader(filename):
        try:
            with io.StringIO(file_contents[filename]) as csv_file:
                reader =  csv.reader(csv_file)
                yield reader
        finally:
            pass
    return mock_loader
