from datetime import date
import numpy as np
import swacmod.historical_nitrate as historical_nitrate
import unittest

class Test_Historical_Nitrate(unittest.TestCase):
	def test_historical_nitrate_returns_zeros_when_disabled(self):
		input_process_enabled = "disabled"
		input_date = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_historical_date = None
		input_historical_mi = None
		input_node = 3

		expected = np.zeros(2)

		self.assert_historical_nitrate(expected, input_process_enabled, input_date, input_historical_date, input_historical_mi, input_node)

	def assert_historical_nitrate(self, expected, input_process_enabled, input_date, input_historical_date, input_historical_mi, input_node):
		data = {
			"params": {
				"historical_nitrate_process": input_process_enabled,
			}, "series" : {
				"date": input_date
			},
		}
		output = {}
		node = input_node
		process_result = historical_nitrate.get_historical_nitrate(data, output, node)
		actual = process_result["historical_nitrate_reaching_water_table_array_tons_per_day"]
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_truncated_historical_nitrate_dates_when_both_sets_of_date_are_empty(self):
		self.assert_calculate_truncated_historical_nitrate_dates([], [], [])

	def assert_calculate_truncated_historical_nitrate_dates(self, expected, input_historical_nitrate_date, input_date):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historical_nitrate_date = input_historical_nitrate_date
		blackboard.date = input_date

		actual = historical_nitrate._calculate_truncated_historical_nitrate_date(blackboard)

		self.assertEqual(expected, actual)
		
	def test_calculate_truncated_historical_nitrate_dates_when_historical_dates_are_empty(self):
		input_historical_nitrate_date = []
		input_date = [date(2023, 1, 3), date(2023, 1, 4), ]
		expected = []
		self.assert_calculate_truncated_historical_nitrate_dates(expected, input_historical_nitrate_date, input_date)

	def test_calculate_truncated_historical_nitrate_dates_when_new_dates_are_empty(self):
		input_historical_nitrate_date = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_date = []
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_dates(expected, input_historical_nitrate_date, input_date)

	def test_calculate_truncated_historical_nitrate_dates_when_historical_dates_are_contiguous_with_new_dates(self):
		input_historical_nitrate_date = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_date = [date(2023, 1, 3), date(2023, 1, 4), ]
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_dates(expected, input_historical_nitrate_date, input_date)

	def test_calculate_truncated_historical_nitrate_dates_when_there_is_a_gap_between_historical_and_new_dates(self):
		input_historical_nitrate_date = [date(2023, 1, 1), date(2023, 1, 2), ]
		input_date = [date(2023, 1, 7), date(2023, 1, 8), ]
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_dates(expected, input_historical_nitrate_date, input_date)

	def test_calculate_truncated_historical_nitrate_dates_when_there_is_an_overlap_between_historical_and_new_dates(self):
		input_historical_nitrate_date = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), ]
		input_date = [date(2023, 1, 3), date(2023, 1, 4),  date(2023, 1, 5),  date(2023, 1, 6), ]
		expected = [date(2023, 1, 1), date(2023, 1, 2), ]
		self.assert_calculate_truncated_historical_nitrate_dates(expected, input_historical_nitrate_date, input_date)

	def test_calculate_historical_nitrate_populates_truncated_historical_nitrate_dates(self):
		actual = historical_nitrate._calculate_historical_nitrate(self.make_sample_blackboard())
		self.assertIsNotNone(actual.truncated_historical_nitrate_dates)

	def make_sample_blackboard(self):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historical_nitrate_date = [date(2023, 1, 1), date(2023, 1, 2), ]
		blackboard.date = [date(2023, 1, 3), date(2023, 1, 4), ]
		return blackboard

	def test_calculate_truncated_historical_mi_array_kg_per_day_does_not_truncate_when_dates_are_not_truncated(self):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.truncated_historical_nitrate_dates = [date(2023, 1, 1), date(2023, 1, 2), ]
		blackboard.historical_mi_array_kg_per_day = np.array([10.0, 20.0])

		actual = historical_nitrate._calculate_truncated_historical_mi_array_kg_per_day(blackboard)

		expected = np.array([10.0, 20.0])
		np.testing.assert_array_equal(expected, actual)

	def test_calculate_truncated_historical_mi_array_kg_per_day_does_not_truncate_when_dates_are_longer_than_mi(self):
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.truncated_historical_nitrate_dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), ]
		blackboard.historical_mi_array_kg_per_day = np.array([10.0, 20.0])

		actual = historical_nitrate._calculate_truncated_historical_mi_array_kg_per_day(blackboard)

		expected = np.array([10.0, 20.0])
		np.testing.assert_array_equal(expected, actual)
