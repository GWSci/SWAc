import unittest
import numpy as np
from swacmod.input_files.input_files_version_2.input_data import ensure_precision

class Test_Precision_Is_Converted_To_64bit(unittest.TestCase):
    def test_floats_are_converted_to_64bit(self):
        data = make_data(np.float32)
        data = ensure_precision(data)
        for key in data['params'].keys():
            self.assertTrue(check_type(data['params'], key, float))
        



def make_data(t_type):
    return {'params': {'number': t_type(1.),
                       'list': [t_type(1.), t_type(2.), t_type(3.)],
                       'dict': {t_type(1.), t_type(2.), t_type(3.)},
                       'tuple': (t_type(1.), t_type(2.), t_type(3.)),
                       'set': set((t_type(1.), t_type(2.), t_type(3.))),
                       'array': np.array([1., 2., 3.], dtype=t_type)
                       }
            }

def check_type(params, key, t_type):
    if key == 'number':
        return isinstance(params[key], t_type)
    elif key == 'list' or key == 'dict' or key == 'tuple' or key == 'set':
        return all([isinstance(val, t_type) for val in params[key]])
    elif key == 'array':
        return params[key].dtype == 'float64'