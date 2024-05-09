import unittest
import numpy as np
import swacmod.snow_melt as snow_melt

class TransmissivityTest(unittest.TestCase):
	def test_zeros(self):
		actual = self.atmospheric_emissivity([0], [0])
		np.testing.assert_almost_equal(0.72, actual)

	def test_non_zero_airtemp(self):
		actual = self.atmospheric_emissivity([3], [0])
		np.testing.assert_almost_equal(0.735, actual)

	def test_cloudiness_1(self):
		actual = self.atmospheric_emissivity([0], [1])
		np.testing.assert_almost_equal(0.9552, actual)

	def test_cloudiness_half(self):
		actual = self.atmospheric_emissivity([0], [0.5])
		np.testing.assert_almost_equal(0.8376, actual)

	def test_both_non_zero(self):
		actual = self.atmospheric_emissivity([15], [0.7])
		np.testing.assert_almost_equal(0.91554, actual)

	def test_lists(self):
		actual = self.atmospheric_emissivity([15, 5, -2, 20], [0.1, 0.9, 0.8, 0.3])
		np.testing.assert_almost_equal([0.81222, 0.93778, 0.90488, 0.86536], actual)

	def atmospheric_emissivity(self, airtemp, cloudiness):
		return snow_melt.AtmosphericEmissivity(np.array(airtemp), np.array(cloudiness))