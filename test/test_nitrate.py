from datetime import date
import unittest
import numpy as np
import os
from swacmod import compile_model
import swacmod.model as m
import swacmod.nitrate as nitrate
import swacmod.timer as timer

class Test_Nitrate(unittest.TestCase):
	def test_calculate_daily_HER(self):
		input_rainfall_ts = np.array([110.0, 220.0, 330.0])
		input_ae = np.array([10.0, 20.0, 30.0])
		expected = np.array([100.0, 200.0, 300.0])
		self.assert_her(input_rainfall_ts, input_ae, expected)

	def test_calculate_daily_HER_can_be_zero(self):
		input_rainfall_ts = np.array([110.0, 0.0, 330.0])
		input_ae = np.array([10.0, 0.0, 330.0])
		expected = np.array([100.0, 0.0, 0.0])
		self.assert_her(input_rainfall_ts, input_ae, expected)

	def test_calculate_daily_HER_cannot_be_less_than_zero(self):
		input_rainfall_ts = np.array([110.0, -2.0, 0.0, 3.0])
		input_ae = np.array([10.0, 0.0, 1.0, 5.0])
		expected = np.array([100.0, 0.0, 0.0, 0])
		self.assert_her(input_rainfall_ts, input_ae, expected)

	def assert_her(self, input_rainfall_ts, input_ae, expected):
		data = None
		output = {
			'rainfall_ts': input_rainfall_ts,
			'ae': input_ae,
		}
		node = None
		actual = nitrate._calculate_her_array_mm_per_day(data, output, node)

		np.testing.assert_array_equal(expected, actual)

	def test_cumulative_fraction_leaked_per_year(self):
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 10.0, 0.05)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 70.0, 0.32)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 110.0, 0.5)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 200.0, 0.7025)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 310.0, 0.95)

	def test_cumulative_fraction_leaked_per_year_can_be_more_than_1(self):
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 376.666666, 1.1)

	def test_cumulative_fraction_leaked_per_year_when_gradient_is_zero(self):
		self.assert_cumulative_fraction_leaked_per_year(110.0, 110.0, 110.0, 376.666666, 0.5)

	def test_cumulative_fraction_leaked_per_year_cannot_be_less_than_than_0(self):
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 5, 0.03)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 4, 0.02)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 3, 0.01)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 2, 0.00)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 1, 0.00)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 0, 0.00)

	def assert_cumulative_fraction_leaked_per_year(self, her_at_5_percent, her_at_50_percent, her_at_95_percent, her, expected):
		actual = m._cumulative_fraction_leaked_per_year(
				her_at_5_percent, her_at_50_percent, her_at_95_percent, her)
		self.assertAlmostEqual(expected, actual)

	def test_cumulative_fraction_leaked_per_day(self):
		her_at_5_percent = 10.0
		her_at_50_percent = 110.0
		her_at_95_percent = 310.0

		testee = lambda her: m._cumulative_fraction_leaked_per_day(
				her_at_5_percent, her_at_50_percent, her_at_95_percent, her)

		self.assertAlmostEqual(0.05 / 365.25, testee(10.0 / 365.25))
		self.assertAlmostEqual(0.32 / 365.25, testee(70.0 / 365.25))
		self.assertAlmostEqual(0.5 / 365.25, testee(110.0 / 365.25))
		self.assertAlmostEqual(0.7025 / 365.25, testee(200.0 / 365.25))
		self.assertAlmostEqual(0.95 / 365.25, testee(310.0 / 365.25))

	def test_calculate_total_mass_leached_from_cell_on_days(self):
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([], testee([], np.array([])))
		np.testing.assert_array_equal([2000.0], testee([date(2023, 1, 1)], np.array([20.0])))
		np.testing.assert_array_equal([2000.0, 8000.0,], testee([date(2023, 1, 1), date(2023, 1, 2)], np.array([20.0, 80.0])))

	def test_calculate_total_mass_leached_from_cell_on_days_limits_by_max_load_for_the_year(self):
		max_load_per_year = 10000 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], np.array([150 * 365.25])))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0],
			testee([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)], np.array([60 * 365.25, 60 * 365.25, 60 * 365.25])))

	def test_calculate_total_mass_leached_from_cell_on_days_resets_limit_on_1st_october(self):
		max_load_per_year = 10000 * 365.25
		her_for_60_percent = 60 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], np.array([150 * 365.25])))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0, 0.6 * max_load_per_year, 0.4 * max_load_per_year, 0],
			testee(
				[date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
				np.array([her_for_60_percent, her_for_60_percent, her_for_60_percent, her_for_60_percent, her_for_60_percent, her_for_60_percent])))
	
	def test_calculate_m0_array_kg_per_day(self):
		max_load_per_year = 10000 * 365.25 * 4
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		data = {
			"time_switcher": timer.make_time_switcher(),
			"series": {
				"date" : [date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)]
			}, "params": {
				"node_areas" : {
					3: 2500
				}, "nitrate_loading" : {
					# Node,UNIQUE,X,Y,LOAD0,HER_5_MaxL,HER_50_Max,HER_95_Max,5PercLoadM,50PercLoad,95PercLoad
					3: [0, 0, 0, max_load_per_year, her_at_5_percent, her_at_50_percent, her_at_95_percent, 0, 0, 0]
				}
			},
		}
		output = None
		node = 3
		her_array_mm_per_day = np.array([60 * 365.25, 60 * 365.25, 60 * 365.25, 60 * 365.25, 60 * 365.25, 60 * 365.25])

		max_load_per_cell_per_year = 10000 * 365.25
		expected = [0.6 * max_load_per_cell_per_year, 0.4 * max_load_per_cell_per_year, 0, 0.6 * max_load_per_cell_per_year, 0.4 * max_load_per_cell_per_year, 0]

		actual = nitrate._calculate_m0_array_kg_per_day(data, output, node, her_array_mm_per_day)

		np.testing.assert_array_equal(expected, actual)		

	def test_calculate_m1_array_kg_per_day(self):
		data = None
		output = {
			"perc_through_root" : np.array([10.0, 10.0])
		}
		node = None
		her_array_mm_per_day = np.array([40.0, 0.0])
		m0_kg_per_day = np.array([1000.0, 1000.0])

		expected = np.array([250.0, 0.0])

		actual = nitrate._calculate_m1_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_kg_per_day)
		np.testing.assert_array_equal(expected, actual)	

	def test_calculate_m1a_array_kg_per_day_for_just_one_day(self):
		input_interflow_volume = np.array([40.0])
		input_infiltration_recharge = np.array([20.0])
		input_interflow_to_rivers = np.array([40.0])
		input_m1_array_kg_per_day = np.array([12.0])
		expected = np.array([2.4])
		self.assert_m1a_array(input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected)

	def test_calculate_m1a_array_kg_per_day_for_three_days_with_accumulation_in_MiT(self):
		input_interflow_volume = np.array([40.0, 35.0, 90.0])
		input_infiltration_recharge = np.array([20.0, 30.0, 10.0])
		input_interflow_to_rivers = np.array([40.0, 35.0, 0.0])
		input_m1_array_kg_per_day = np.array([12, 25.2, 39.5])
		expected = np.array([2.4, 9.0, 5.0])
		self.assert_m1a_array(input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected)
	
	def test_calculate_m1a_array_kg_per_day_when_interflow_store_components_equal_zero(self):
		input_interflow_volume = np.array([0.0, 0.0, 40.0])
		input_infiltration_recharge = np.array([0.0, 0.0, 20.0])
		input_interflow_to_rivers = np.array([0.0, 0.0, 40.0])
		input_m1_array_kg_per_day = np.array([4.0, 4.0, 4.0])
		expected = np.array([0.0, 0.0, 2.4])
		self.assert_m1a_array(input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected)

	def assert_m1a_array(self, input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected):
		data = {
			"time_switcher": timer.make_time_switcher(),
		}
		output = {
			"interflow_volume" : input_interflow_volume,
			"infiltration_recharge" : input_infiltration_recharge,
			"interflow_to_rivers" : input_interflow_to_rivers,
		}
		node = None

		actual = nitrate._calculate_m1a_array_kg_per_day(data, output, node, input_m1_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_m2_array_kg_per_day(self):
		data = None
		output = {
			"runoff_recharge" : np.array([100.0, 0.0, 0.0]),
			"macropore_att" : np.array([0.0, 40.0, 40.0]),
			"macropore_dir" : np.array([0.0, 60.0, 60.0]),
		}
		node = None
		her_array_mm_per_day = np.array([10.0, 20.0, 0.0])
		m0_array_kg_per_day = np.array([50.0, 60.0, 60.0])

		actual = nitrate._calculate_m2_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([500.0, 300.0, 0.0])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_m3_array_kg_per_day(self):
		data = None
		output = {
			"rapid_runoff" : np.array([100.0, 100.0]),
			"runoff_recharge" : np.array([45.0, 45.0]),
		}
		node = None
		her_array_mm_per_day = np.array([11.0, 0.0])
		m0_array_kg_per_day = np.array([7.0, 7.0])
		
		actual = nitrate._calculate_m3_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([35.0, 0.0])

		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_mi_array_kg_per_day(self):
		m1a_array_kg_per_day = np.array([100.0, 200.0, 300.0])
		m2_array_kg_per_day = np.array([40.0, 50.0, 60.0])

		actual = nitrate._calculate_mi_array_kg_per_day(m1a_array_kg_per_day, m2_array_kg_per_day)
		expected = np.array([140.0, 250.0, 360.0])

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
			"proportion_100" : None,
			"time_switcher": timer.make_time_switcher(),
			"params": {
				"node_areas": {7: 2500.0},
				"nitrate_calibration_a": 1.38,
				"nitrate_calibration_mu": 1.58,
				"nitrate_calibration_sigma": 3.96,
				"nitrate_calibration_mean_hydraulic_conductivity": 1.7,
				"nitrate_calibration_mean_velocity_of_unsaturated_transport": 0.0029,
				"nitrate_depth_to_water": {7: [0.00205411]},
				"nitrate_loading": {7: [0, 0, 0, max_load_per_year_kg_per_hectare, her_at_5_percent, her_at_50_percent, her_at_95_percent]},
				"nitrate_process": "enabled",
			}, "series" : {
				"date": [date(2023, 1, 1), date(2023, 1, 2), ]
			},
		}
		output = {
			"rainfall_ts": np.array([130.0, 130.0]),
			"ae": np.array([50.0, 50.0]),
			"perc_through_root": np.array([40.0, 40.0]),
			"interflow_volume": np.array([2.0, 20.0]),
			"infiltration_recharge": np.array([1.0, 9.0]),
			"interflow_to_rivers": np.array([2.0, 6.0]),
			"runoff_recharge": np.array([8.0, 8.0]),
			"macropore_att": np.array([4.0, 4.0]),
			"macropore_dir": np.array([4.0, 4.0]),
			"rapid_runoff": np.array([32.0, 32.0]),
			"combined_recharge": np.array([5, 5]),
		}
		node = 7
		actual = nitrate.calculate_nitrate(data, output, node)
		np.testing.assert_array_almost_equal(np.array([80.0, 80.0]), actual["her_array_mm_per_day"])
		np.testing.assert_array_almost_equal(np.array([100.0, 100.0]), actual["m0_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([50.0, 50.0]), actual["m1_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([10.0, 18.0]), actual["m1a_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([20.0, 20.0]), actual["m2_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([30.0, 30.0]), actual["m3_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([30.0, 38.0]), actual["mi_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.6]), actual["proportion_reaching_water_table_array_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 18.0]), actual["nitrate_reaching_water_table_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.018]), actual["nitrate_reaching_water_table_array_tons_per_day"])

	def test_calculate_nitrate_when_disabled(self):
		max_load_per_year_kg_per_hectare = 1000
		her_at_5_percent = 10
		her_at_50_percent = 100
		her_at_95_percent = 190

		data = {
			"proportion_100" : None,
			"time_switcher": timer.make_time_switcher(),
			"params": {
				"node_areas": {7: 2500.0},
				"nitrate_depth_to_water": {7: [0.00205411]},
				"nitrate_loading": {7: [0, 0, 0, max_load_per_year_kg_per_hectare, her_at_5_percent, her_at_50_percent, her_at_95_percent]},
				"nitrate_process": "disabled",
			}, "series" : {
				"date": [date(2023, 1, 1), date(2023, 1, 2), ]
			},
		}
		output = {
			"rainfall_ts": np.array([130.0, 130.0]),
			"ae": np.array([50.0, 50.0]),
			"perc_through_root": np.array([40.0, 40.0]),
			"interflow_volume": np.array([1.0, 1.0]),
			"infiltration_recharge": np.array([2.0, 2.0]),
			"interflow_to_rivers": np.array([2.0, 2.0]),
			"runoff_recharge": np.array([8.0, 8.0]),
			"macropore_att": np.array([4.0, 4.0]),
			"macropore_dir": np.array([4.0, 4.0]),
			"rapid_runoff": np.array([32.0, 32.0]),
			"combined_recharge": np.array([5, 5]),
		}
		node = 7
		actual = nitrate.calculate_nitrate(data, output, node)
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["her_array_mm_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["m0_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["m1_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["m1a_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["m2_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["m3_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["mi_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["proportion_reaching_water_table_array_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["nitrate_reaching_water_table_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["nitrate_reaching_water_table_array_tons_per_day"])

	def test_output_file_path(self):
		data = {
			"params" : {
				"run_name" : "aardvark",
			}
		}
		expected = ""
		actual = nitrate.make_output_filename(data)
		sep = os.path.sep
		expected = "output_files" + sep + "aardvark_nitrate.csv"
		isPassed = actual.endswith(expected)
		message = f"Expected '{actual}' to end with '{expected}'."
		self.assertTrue(isPassed, message)

def calculate_total_mass_leached_for_test(days, her_per_day):
		max_load_per_year = 10000 * 365.25
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		return m._calculate_total_mass_leached_from_cell_on_days(
			max_load_per_year,
			her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			days,
			her_per_day)
