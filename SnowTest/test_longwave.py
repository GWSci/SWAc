import unittest
import numpy as np
import snow_melt

class LongwaveTest(unittest.TestCase):
	def test_Longwave(self):
		emissivity = np.array([0.8, 0.3])
		temp = np.array([10, 15])
		expected = [25197.2027682211, 10134.2550589769]
		actual = snow_melt.Longwave(emissivity, temp)
		np.testing.assert_almost_equal(expected, actual)
