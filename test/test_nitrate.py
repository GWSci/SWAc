import unittest
import numpy as np
import swacmod.nitrate as nitrate

class Test_Nitrate(unittest.TestCase):
	def test_calculate_daily_HER(self):
		data = None
		output = {
			'rainfall_ts': np.array([110, 220, 330]),
			'ae': np.array([10, 20, 30]),
		}
		node = None
		actual = nitrate._calculate_daily_HER(data, output, node)

		expected = np.array([100, 200, 300])
		np.testing.assert_array_equal(expected, actual)

	def test_cumulative_fraction_leaked_per_year(self):
		her_at_5_percent = 10.0
		her_at_50_percent = 110.0
		her_at_95_percent = 210.0
		max_load = 10000

		testee = lambda her: cumulative_fraction_leaked_per_year(
				max_load, her_at_5_percent, her_at_50_percent, her_at_95_percent, her)

		self.assertAlmostEqual(500, testee(10.0))
		self.assertAlmostEqual(5000, testee(110.0))

def cumulative_fraction_leaked_per_year(
		max_load,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		her):
	# y = mx + c
	m = 0.45 / (her_at_50_percent - her_at_5_percent)
	c = 0.5 - (her_at_50_percent * m)
	x = her
	y = (m * x) + c
	percentage = y
	return max_load * percentage
