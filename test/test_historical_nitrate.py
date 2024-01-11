from datetime import date
import numpy as np
import swacmod.historical_nitrate as historical_nitrate
import swacmod.nitrate as nitrate
import unittest

class Test_Historical_Nitrate(unittest.TestCase):
	def test_historical_nitrate_returns_zeros_when_disabled(self):
		input_process_enabled = "disabled"
		input_days = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_historical_days = None
		input_historical_mi = None
		input_node = 3

		expected = np.zeros(2)

		self.assert_historical_nitrate(expected, input_process_enabled, input_days, input_historical_days, input_historical_mi, input_node)

	def assert_historical_nitrate(self, expected, input_process_enabled, input_days, input_historical_days, input_historical_mi, input_node):
		data = {
			"params": {
				"historical_nitrate_process": input_process_enabled,
			}, "series" : {
				"date": input_days
			},
		}
		output = {}
		node = input_node
		process_result = historical_nitrate.get_historical_nitrate(data, output, node)
		actual = process_result["historical_nitrate_reaching_water_table_array_tons_per_day"]
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_truncated_historical_nitrate_days_when_both_sets_of_date_are_empty(self):
		self.assert_calculate_truncated_historical_nitrate_days([], [], [])

	def assert_calculate_truncated_historical_nitrate_days(self, expected, input_historical_nitrate_days, input_days):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historical_nitrate_days = input_historical_nitrate_days
		blackboard.days = input_days

		actual = historical_nitrate._calculate_truncated_historical_nitrate_days(blackboard)

		self.assertEqual(expected, actual)
		
	def test_calculate_truncated_historical_nitrate_dates_when_historical_dates_are_empty(self):
		input_historical_nitrate_days = []
		input_days = [date(2023, 1, 3), date(2023, 1, 4), ]
		expected = []
		self.assert_calculate_truncated_historical_nitrate_days(expected, input_historical_nitrate_days, input_days)

	def test_calculate_truncated_historical_nitrate_dates_when_new_dates_are_empty(self):
		input_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_days = []
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_days(expected, input_historical_nitrate_days, input_days)

	def test_calculate_truncated_historical_nitrate_dates_when_historical_dates_are_contiguous_with_new_dates(self):
		input_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_days = [date(2023, 1, 3), date(2023, 1, 4), ]
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_days(expected, input_historical_nitrate_days, input_days)

	def test_calculate_truncated_historical_nitrate_dates_when_there_is_a_gap_between_historical_and_new_dates(self):
		input_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_days = [date(2023, 1, 7), date(2023, 1, 8), ]
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_days(expected, input_historical_nitrate_days, input_days)

	def test_calculate_truncated_historical_nitrate_dates_when_there_is_an_overlap_between_historical_and_new_dates(self):
		input_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), ]
		input_days = [date(2023, 1, 3), date(2023, 1, 4),  date(2023, 1, 5),  date(2023, 1, 6), ]
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_days(expected, input_historical_nitrate_days, input_days)

	def test_calculate_historical_nitrate_populates_truncated_historical_nitrate_dates(self):
		actual = historical_nitrate._calculate_historical_nitrate(self.make_sample_blackboard())
		self.assertIsNotNone(actual.truncated_historical_nitrate_days)

	def test_calculate_historical_nitrate_populates_truncated_historical_mi_array_kg_per_day(self):
		actual = historical_nitrate._calculate_historical_nitrate(self.make_sample_blackboard())
		self.assertIsNotNone(actual.truncated_historical_mi_array_kg_per_day)

	def test_calculate_historical_nitrate_populates_historic_proportion_reaching_water_table_array_per_day(self):
		actual = historical_nitrate._calculate_historical_nitrate(self.make_sample_blackboard())
		self.assertIsNotNone(actual.historic_proportion_reaching_water_table_array_per_day)

	def test_calculate_historical_nitrate_populates_historical_mass_reaching_water_table_array_kg_per_day(self):
		actual = historical_nitrate._calculate_historical_nitrate(self.make_sample_blackboard())
		self.assertIsNotNone(actual.historical_mass_reaching_water_table_array_kg_per_day)

	def test_calculate_historical_nitrate_populates_historical_nitrate_reaching_water_table_array_tons_per_day(self):
		actual = historical_nitrate._calculate_historical_nitrate(self.make_sample_blackboard())
		self.assertIsNotNone(actual.historical_nitrate_reaching_water_table_array_tons_per_day)

	def make_sample_blackboard(self):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.days = [date(2023, 1, 3), date(2023, 1, 4), ]
		blackboard.historical_mi_array_kg_per_day = np.array([10.0, 20.0])
		blackboard.historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), ]
		blackboard.nitrate_depth_to_water = np.array([10.0])
		blackboard.mean_hydraulic_conductivity = 1.0
		blackboard.mean_velocity_of_unsaturated_transport = 1.0
		blackboard.a = 10.0
		blackboard.μ = 0.0
		blackboard.σ = 1.0
		return blackboard

	def test_calculate_truncated_historical_mi_array_kg_per_day_does_not_truncate_when_dates_are_not_truncated(self):
		expected = np.array([10.0, 20.0])
		truncated_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), ]
		historical_mi_array_kg_per_day = [10.0, 20.0]
		self.assert_calculate_truncated_historical_mi_array_kg_per_day(
			expected,
			truncated_historical_nitrate_days,
			historical_mi_array_kg_per_day)

	def test_calculate_truncated_historical_mi_array_kg_per_day_does_not_truncate_when_dates_are_longer_than_mi(self):
		expected = [10.0, 20.0]
		truncated_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), ]
		historical_mi_array_kg_per_day = [10.0, 20.0]
		self.assert_calculate_truncated_historical_mi_array_kg_per_day(
			expected,
			truncated_historical_nitrate_days,
			historical_mi_array_kg_per_day)

	def test_calculate_truncated_historical_mi_array_kg_per_day_does_not_truncate_when_dates_are_shorter_than_mi(self):
		expected = [10.0, 20.0]
		truncated_historical_nitrate_days = [date(2023, 1, 1), date(2023, 1, 2)]
		historical_mi_array_kg_per_day = [10.0, 20.0, 30.0, 40.0]
		self.assert_calculate_truncated_historical_mi_array_kg_per_day(
			expected,
			truncated_historical_nitrate_days,
			historical_mi_array_kg_per_day)

	def assert_calculate_truncated_historical_mi_array_kg_per_day(
			self,
			expected,
			input_truncated_historical_nitrate_days,
			input_historical_mi_array_kg_per_day
	):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.truncated_historical_nitrate_days = input_truncated_historical_nitrate_days
		blackboard.historical_mi_array_kg_per_day = np.array(input_historical_mi_array_kg_per_day)

		actual = historical_nitrate._calculate_truncated_historical_mi_array_kg_per_day(blackboard)
		np.testing.assert_array_equal(np.array(expected), actual)

	def test__calculate_historic_proportion_reaching_water_table_array_per_day(self):
		historic_days = [
			date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
		new_days = [
			date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6),
			date(2023, 1, 7), date(2023, 1, 8), date(2023, 1, 9), date(2023, 1, 10)]
		combined_days = [
			date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
			date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6),
			date(2023, 1, 7), date(2023, 1, 8), date(2023, 1, 9), date(2023, 1, 10)]
		
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		self.assign_common_blackboard_inputs_for_proportion_reaching_water_table(blackboard)
		blackboard.days = new_days
		blackboard.historical_nitrate_days = historic_days
		actual = historical_nitrate._calculate_historic_proportion_reaching_water_table_array_per_day(blackboard)

		cumulative_proportion_for_the_entire_period = self.make_reference_proportion_reaching_water_table_for_combined_historic_and_new_periods(combined_days)
		expected = cumulative_proportion_for_the_entire_period[3:]
		np.testing.assert_allclose(expected, actual)

	def assign_common_blackboard_inputs_for_proportion_reaching_water_table(self, blackboard):
		blackboard.nitrate_depth_to_water = np.array([10.0])
		blackboard.mean_hydraulic_conductivity = 1.0
		blackboard.mean_velocity_of_unsaturated_transport = 1.0
		blackboard.a = 10.0
		blackboard.μ = 0.0
		blackboard.σ = 1.0

	def make_reference_proportion_reaching_water_table_for_combined_historic_and_new_periods(self, combined_days):
		nitrate_blackboard = nitrate.NitrateBlackboard()
		self.assign_common_blackboard_inputs_for_proportion_reaching_water_table(nitrate_blackboard)
		nitrate_blackboard.days = combined_days
		cumulative_proportion_for_the_entire_period = nitrate._calculate_proportion_reaching_water_table_array_per_day(nitrate_blackboard)
		np.testing.assert_allclose(
			np.array([False, True, True, True, True, True, True, True, True, True]),
			cumulative_proportion_for_the_entire_period > 0
		)
		return cumulative_proportion_for_the_entire_period

	def test_calculate_historical_mass_reaching_water_table_array_kg_per_day(self):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historic_proportion_reaching_water_table_array_per_day = np.array([0.1, 0.2, 0.3])
		blackboard.truncated_historical_mi_array_kg_per_day = np.array([10.0, 100.0, 1000.0])

		actual = historical_nitrate._calculate_historical_mass_reaching_water_table_array_kg_per_day(blackboard)

		expected = np.array([1.0, 12.0, 123.0])
		np.testing.assert_allclose(expected, actual)

	def test_convert_kg_to_tons_array(self):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historical_mass_reaching_water_table_array_kg_per_day = np.array([1000.0, 2000.0, 3000.0])

		actual = historical_nitrate._convert_kg_to_tons_array(blackboard)

		expected = np.array([1.0, 2.0, 3.0])
		np.testing.assert_allclose(expected, actual)

	def test_historical_nitrate_initialise_blackboard(self):
		data, output, node = self.make_sample_data_output_node()
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()

		actual = blackboard.initialise_blackboard(data, output, node)

		expected = self.make_sample_blackboard()
		self.assertEqual(expected.a, actual.a)
		self.assertEqual(expected.days, actual.days)
		np.testing.assert_allclose(expected.historical_mi_array_kg_per_day, actual.historical_mi_array_kg_per_day)
		self.assertEqual(expected.historical_nitrate_days, actual.historical_nitrate_days)
		np.testing.assert_allclose(expected.nitrate_depth_to_water, actual.nitrate_depth_to_water)
		self.assertEqual(expected.mean_hydraulic_conductivity, actual.mean_hydraulic_conductivity)
		self.assertEqual(expected.mean_velocity_of_unsaturated_transport, actual.mean_velocity_of_unsaturated_transport)
		self.assertEqual(expected.μ, actual.μ)
		self.assertEqual(expected.σ, actual.σ)

	def make_sample_data_output_node(self):
		data = {
			"params": {
				"nitrate_calibration_a": 10.0,
				"nitrate_depth_to_water": {7: np.array([10.0])},
				"nitrate_calibration_mean_hydraulic_conductivity" : 1.0,
				"nitrate_calibration_mean_velocity_of_unsaturated_transport" : 1.0,
				"nitrate_calibration_mu": 0.0,
				"nitrate_calibration_sigma": 1.0,
			},
			"series": {
				"date": [date(2023, 1, 3), date(2023, 1, 4)],
				"historical_mi_kg_per_day" : {7 : np.array([10.0, 20.0])},
				"historical_nitrate_days" : [date(2023, 1, 1), date(2023, 1, 2)],
			}
		}
		output = {}
		node = 7
		return data, output, node

	def test_get_historical_nitrate_return_answer_when_enabled(self):
		data, output, node = self.make_sample_data_output_node()
		actual = historical_nitrate.get_historical_nitrate(data, output, node)