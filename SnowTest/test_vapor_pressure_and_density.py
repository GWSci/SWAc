import unittest
import numpy as np
import snow_melt

class TransmissivityTest(unittest.TestCase):
	def test_SatVaporPressure(self):
		T_C = np.array([0, -5, 10, 15])
		expected = [0.6110000, 0.4209749, 1.2302038, 1.7096471]
		actual = snow_melt.SatVaporPressure(T_C)
		np.testing.assert_almost_equal(expected, actual)
		
	def test_SatVaporDensity(self):
		T_C = np.array([0, -5, 10, 15])
		expected = [00.0048, 0.0034, 0.0094, 0.0128]
		actual = snow_melt.SatVaporDensity(T_C)
		np.testing.assert_almost_equal(expected, actual)
