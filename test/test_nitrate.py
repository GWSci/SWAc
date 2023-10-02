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
		her_at_95_percent = 310.0

		testee = lambda her: cumulative_fraction_leaked_per_year(
				her_at_5_percent, her_at_50_percent, her_at_95_percent, her)

		self.assertAlmostEqual(0.05, testee(10.0))
		self.assertAlmostEqual(0.32, testee(70.0))
		self.assertAlmostEqual(0.5, testee(110.0))
		self.assertAlmostEqual(0.7025, testee(200.0))
		self.assertAlmostEqual(0.95, testee(310.0))

	def test_cumulative_fraction_leaked_per_day(self):
		her_at_5_percent = 10.0
		her_at_50_percent = 110.0
		her_at_95_percent = 310.0

		testee = lambda her: cumulative_fraction_leaked_per_day(
				her_at_5_percent, her_at_50_percent, her_at_95_percent, her)

		self.assertAlmostEqual(0.05 / 365.25, testee(10.0 / 365.25))
		self.assertAlmostEqual(0.32 / 365.25, testee(70.0 / 365.25))
		self.assertAlmostEqual(0.5 / 365.25, testee(110.0 / 365.25))
		self.assertAlmostEqual(0.7025 / 365.25, testee(200.0 / 365.25))
		self.assertAlmostEqual(0.95 / 365.25, testee(310.0 / 365.25))

	def test_calculate_total_mass_leached_from_cell_on_days(self):
		max_load_per_year = 10000 * 365.25
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		testee = lambda days, her_per_day: calculate_total_mass_leached_from_cell_on_days(
				max_load_per_year,
				her_at_5_percent,
				her_at_50_percent,
				her_at_95_percent,
				days,
				her_per_day)
		
		np.testing.assert_array_equal([], testee([], []))

def calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_per_day):
	pass
	
def cumulative_fraction_leaked_per_year(
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		her):
	x = her
	is_below_50_percent = her < her_at_50_percent
	upper = her_at_50_percent if is_below_50_percent else her_at_95_percent
	lower = her_at_5_percent if is_below_50_percent else her_at_50_percent
	# y = mx + c
	m = 0.45 / (upper - lower)
	c = 0.5 - (her_at_50_percent * m)
	y = (m * x) + c
	return y

def cumulative_fraction_leaked_per_day(
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		her_per_day):
	days_in_year = 365.25
	her_per_year = days_in_year * her_per_day
	x = her_per_year
	is_below_50_percent = her_per_year < her_at_50_percent
	upper = her_at_50_percent if is_below_50_percent else her_at_95_percent
	lower = her_at_5_percent if is_below_50_percent else her_at_50_percent
	# y = mx + c
	m = 0.45 / (upper - lower)
	c = 0.5 - (her_at_50_percent * m)
	y = (m * x) + c
	return y / days_in_year
