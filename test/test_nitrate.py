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
		actual = nitrate._calculate_her_array_mm_per_day(data, output, node)

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
	
	def test_calculate_m0_array_kg_per_day(self):
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

		actual = nitrate._calculate_m0_array_kg_per_day(data, output, node, her_array_mm_per_day)

		np.testing.assert_array_equal(expected, actual)		

	def test_calculate_m1_array_kg_per_day(self):
		data = None
		output = {
			"perc_through_root" : np.array([10])
		}
		node = None
		her_array_mm_per_day = np.array([40.0])
		m0_kg_per_day = np.array([1000.0])

		expected = np.array([250])

		actual = nitrate._calculate_m1_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_kg_per_day)
		np.testing.assert_array_equal(expected, actual)	

	def test_calculate_m1a_array_kg_per_day_for_just_one_day(self):
		data = None
		output = {
			"interflow_volume" : np.array([20]),
			"infiltration_recharge" : np.array([40]),
			"interflow_to_rivers" : np.array([40]),
		}
		node = None
		m1_array_kg_per_day = np.array([12])

		expected = np.array([2.4])

		actual = nitrate._calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected, actual)	

	def test_calculate_m1a_array_kg_per_day_for_three_days_with_accumulation_in_MiT(self):
		data = None
		output = {
			"interflow_volume" : np.array([20, 30, 10]),
			"infiltration_recharge" : np.array([40, 35, 90]),
			"interflow_to_rivers" : np.array([40, 35, 0]),
		}
		node = None
		m1_array_kg_per_day = np.array([12, 20.4, 29])

		expected = np.array([2.4, 9, 5])

		actual = nitrate._calculate_m1a_array_kg_per_day(data, output, node, m1_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_m2_array_kg_per_day(self):
		data = None
		output = {
			"runoff_recharge" : np.array([100.0, 0.0]),
			"macropore_att" : np.array([0.0, 40.0]),
			"macropore_dir" : np.array([0.0, 60.0]),
		}
		node = None
		her_array_mm_per_day = np.array([10.0, 20.0])
		m0_array_kg_per_day = np.array([50.0, 60.0])

		actual = nitrate._calculate_m2_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([500.0, 300.0])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_m3_array_kg_per_day(self):
		data = None
		output = {
			"rapid_runoff" : np.array([100.0]),
			"runoff_recharge" : np.array([45.0]),
		}
		node = None
		her_array_mm_per_day = np.array([11.0])
		m0_array_kg_per_day = np.array([7.0])
		
		actual = nitrate._calculate_m3_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([35.0])

		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_mi_array_kg_per_day(self):
		m1a_array_kg_per_day = np.array([100.0, 200.0, 300.0])
		m2_array_kg_per_day = np.array([40.0, 50.0, 60.0])

		actual = nitrate._calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day)
		expected = np.array([140.0, 250.0, 360.0])

		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_recharge_concentration_kg_per_m3(self):
		data = None
		output = {
			'combined_recharge': np.array([2.0, 3.0, 5.0]),
		}
		node = None
		mass_reaching_water_table_array_kg_per_day = [14.0, 33.0, 65.0]
		expected = [7.0, 11.0, 13.0]
		actual = nitrate._calculate_recharge_concentration_kg_per_m3(data, output, node, mass_reaching_water_table_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_convert_kg_to_tons_array(self):
		np.testing.assert_array_almost_equal(np.array([]), nitrate._convert_kg_to_tons_array(np.array([])))
		np.testing.assert_array_almost_equal(np.array([1.0]), nitrate._convert_kg_to_tons_array(np.array([1000.0])))
		np.testing.assert_array_almost_equal(np.array([0.5, 1.0, 3.0]), nitrate._convert_kg_to_tons_array(np.array([500, 1000.0, 3000.0])))
	
	def test_calculate_nitrate(self):
		max_load_per_year_kg_per_hectare = 1000
		her_at_5_percent = 10
		her_at_50_percent = 100
		her_at_95_percent = 190

		data = {
			"params": {
				"node_areas": {7: [2500.0]},
				"nitrate_depth_to_water": {7: [100]},
				"nitrate_leaching": {7: [0, 0, 0, max_load_per_year_kg_per_hectare, her_at_5_percent, her_at_50_percent, her_at_95_percent]},
			}, "series" : {
				"date": [date(2023, 1, 1)]
			},
		}
		output = {
			"rainfall_ts": np.array([130.0]),
			"ae": np.array([50.0]),
			"perc_through_root": np.array([40.0]),
			"interflow_volume": np.array([1.0]),
			"infiltration_recharge": np.array([2.0]),
			"interflow_to_rivers": np.array([2.0]),
			"runoff_recharge": np.array([5]),
			"macropore_att": np.array([5]),
			"macropore_dir": np.array([5]),
			"rapid_runoff": np.array([5]),
			"combined_recharge": np.array([5]),
		}
		node = 7
		actual = nitrate.calculate_nitrate(data, output, node)
		np.testing.assert_array_almost_equal(np.array([80.0]), actual["her_array_mm_per_day"])
		np.testing.assert_array_almost_equal(np.array([100.0]), actual["m0_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([50.0]), actual["m1_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([10.0]), actual["m1a_array_kg_per_day"])

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
