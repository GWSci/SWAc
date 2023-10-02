from datetime import date
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
		actual = nitrate._calculate_her_mm_per_day(data, output, node)

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
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([], testee([], []))
		np.testing.assert_array_equal([2000.0], testee([date(2023, 1, 1)], [20.0]))
		np.testing.assert_array_equal([2000.0, 8000.0,], testee([date(2023, 1, 1), date(2023, 1, 2)], [20.0, 80.0]))

	def test_calculate_total_mass_leached_from_cell_on_days_limits_by_max_load_for_the_year(self):
		max_load_per_year = 10000 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], [150 * 365.25]))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0],
			testee([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)], [60 * 365.25, 60 * 365.25, 60 * 365.25]))

	def test_calculate_total_mass_leached_from_cell_on_days_resets_limit_on_1st_october(self):
		max_load_per_year = 10000 * 365.25
		her_for_60_percent = 60 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], [150 * 365.25]))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0, 0.6 * max_load_per_year, 0.4 * max_load_per_year, 0],
			testee(
				[date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
				[her_for_60_percent] * 6))
	
	def test_calculate_m0_kg_per_day(self):
		max_load_per_year = 10000 * 365.25 * 4
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		data = {
			"series": {
				"date" : [date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)]
			}, "params": {
				"node_areas" : {
					3: [2500]
				}, "nitrate_leaching" : {
					# Node,UNIQUE,X,Y,LOAD0,HER_5_MaxL,HER_50_Max,HER_95_Max,5PercLoadM,50PercLoad,95PercLoad
					3: [0, 0, 0, max_load_per_year, her_at_5_percent, her_at_50_percent, her_at_95_percent, 0, 0, 0]
				}
			},
		}
		output = None
		node = 3
		her_array_mm_per_day = [60 * 365.25] * 6

		max_load_per_cell_per_year = 10000 * 365.25
		expected = [0.6 * max_load_per_cell_per_year, 0.4 * max_load_per_cell_per_year, 0, 0.6 * max_load_per_cell_per_year, 0.4 * max_load_per_cell_per_year, 0]

		actual = calculate_m0_kg_per_day(data, output, node, her_array_mm_per_day)

		np.testing.assert_array_equal(expected, actual)

def calculate_m0_kg_per_day(data, output, node, her_array_mm_per_day):
	params = data["params"]
	cell_area_m_sq = params["node_areas"][node][0]
	days = data["series"]["date"]

	nitrate_leaching = params["nitrate_leaching"][node]
	max_load_per_year_kg_per_hectare = nitrate_leaching[3]
	her_at_5_percent = nitrate_leaching[4]
	her_at_50_percent = nitrate_leaching[5]
	her_at_95_percent = nitrate_leaching[6]

	hectare_area_m_sq = 10000
	max_load_per_year_kg_per_cell = max_load_per_year_kg_per_hectare * cell_area_m_sq / hectare_area_m_sq
	
	m0_array_kg_per_day = calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_array_mm_per_day)
	return m0_array_kg_per_day
		

def calculate_total_mass_leached_for_test(days, her_per_day):
		max_load_per_year = 10000 * 365.25
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		return calculate_total_mass_leached_from_cell_on_days(
			max_load_per_year,
			her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			days,
			her_per_day)

def calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_per_day):
	length = len(days)
	result = np.zeros(length)
	remaining_for_year = max_load_per_year_kg_per_cell
	for i in range(length):
		day = days[i]
		her = her_per_day[i]
		if (day.month == 10) and (day.day == 1):
			remaining_for_year = max_load_per_year_kg_per_cell
		fraction_leached = cumulative_fraction_leaked_per_day(her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			her)
		mass_leached_for_day = min(remaining_for_year, max_load_per_year_kg_per_cell * fraction_leached)
		remaining_for_year -= mass_leached_for_day
		result[i] = mass_leached_for_day
	return result
	
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
