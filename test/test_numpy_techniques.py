import unittest
import numpy as np

class test_numpy_techniques(unittest.TestCase):
    def test_vectorized_macro_act(self):
        zone_mac = 1
        length = 10
        months = np.array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
        macro_act = np.array([[100, 105], [110, 115], [120, 125]])
        indexes = np.arange(length)
        original_method = np.vectorize(lambda num: macro_act[months[num]][zone_mac])(indexes)
        optimised_method = macro_act[months, zone_mac]
        expected = np.array([115, 125, 105, 115, 125, 105, 115, 125, 105, 115])
        np.testing.assert_equal(expected, original_method)
        np.testing.assert_equal(expected, optimised_method)
