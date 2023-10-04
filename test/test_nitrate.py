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

		testee = lambda her: nitrate._cumulative_fraction_leaked_per_year(
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

		testee = lambda her: nitrate._cumulative_fraction_leaked_per_day(
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

		actual = nitrate._calculate_m0_kg_per_day(data, output, node, her_array_mm_per_day)

		np.testing.assert_array_equal(expected, actual)		

	def test_calculate_m1_arr_mm_per_day(self):
		data = None
		output = {
			"perc_through_root" : np.array([10])
		}
		node = None
		her_array_mm_per_day = np.array([40.0])
		m0_kg_per_day = np.array([1000.0])

		expected = np.array([250])

		actual = nitrate._calculate_m1_arr_mm_per_day(data, output, node, her_array_mm_per_day, m0_kg_per_day)
		np.testing.assert_array_equal(expected, actual)	

	def test_calculate_m1a_arr_mm_per_day(self):
		data = None
		output = {
			"interflow_volume" : np.array([20]),
			"infiltration_recharge" : np.array([40]),
			"interflow_to_rivers" : np.array([40]),
		}
		node = None
		m1_arr_kg_per_day = np.array([12])

		expected = np.array([2.4])

		actual = _calculate_m1a_arr_kg_per_day(data, output, node, m1_arr_kg_per_day)
		np.testing.assert_array_almost_equal(expected, actual)	

def _calculate_m1a_arr_kg_per_day(data, output, node, m1_arr_kg_per_day):
	interflow_volume_mm = output["interflow_volume"]
	infiltration_recharge_mm_per_day = output["infiltration_recharge"]
	interflow_to_rivers_mm_per_day = output["interflow_to_rivers"]

	soil_percolation_mm_per_day = interflow_volume_mm + infiltration_recharge_mm_per_day + interflow_to_rivers_mm_per_day
	proportion = interflow_volume_mm / soil_percolation_mm_per_day

	m1a_arr_kg_per_day = m1_arr_kg_per_day * proportion
	return m1a_arr_kg_per_day
	
def calculate_total_mass_leached_for_test(days, her_per_day):
		max_load_per_year = 10000 * 365.25
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		return nitrate._calculate_total_mass_leached_from_cell_on_days(
			max_load_per_year,
			her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			days,
			her_per_day)
