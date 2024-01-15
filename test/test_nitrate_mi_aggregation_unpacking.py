from datetime import date
import swacmod.historical_nitrate as historical_nitrate
import unittest
import numpy as np

class Test_Nitrate_mi_Aggregation_Unpacking(unittest.TestCase):
	def test_nitrate_mi_aggregation_unpacking_for_one_day(self):
		historical_time_periods = {0: [1, 2]}
		expected = [3.0]
		historical_mi_array_kg_per_time_period = {0: np.array([3.0])}
		node = 0
		self.assert_mi_aggregation_unpacking(expected, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def test_nitrate_mi_aggregation_unpacking_for_one_day_on_different_node(self):
		historical_time_periods = {0: [1, 2]}
		expected = [5.0]
		historical_mi_array_kg_per_time_period = {0: np.array([3.0]), 1: np.array([5.0])}
		node = 1
		self.assert_mi_aggregation_unpacking(expected, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def test_nitrate_mi_aggregation_unpacking_for_several_days_in_same_time_period(self):
		historical_time_periods = {0: [1, 4]}
		expected = [5.0, 5.0, 5.0]
		historical_mi_array_kg_per_time_period = {0: np.array([15.0])}
		node = 0
		self.assert_mi_aggregation_unpacking(expected, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def test_nitrate_mi_aggregation_unpacking_for_several_days_each_in_its_own_time_period(self):
		historical_time_periods = {0: [1, 2], 1: [2, 3], 2: [3, 4]}
		expected = [1.0, 20.0, 300.0]
		historical_mi_array_kg_per_time_period = {0: np.array([1.0, 20.0, 300.0])}
		node = 0
		self.assert_mi_aggregation_unpacking(expected, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def test_nitrate_mi_aggregation_unpacking_for_several_time_periods_and_node_0(self):
		historical_time_periods = {0: [1, 3], 1: [3, 6],}
		expected = [5.0, 5.0, 7.0, 7.0, 7.0]
		historical_mi_array_kg_per_time_period = {
			0: np.array([10.0, 21.0]),
			1: np.array([22.0, 39.0]),
		}
		node = 0
		self.assert_mi_aggregation_unpacking(expected, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def test_nitrate_mi_aggregation_unpacking_for_several_time_periods_and_node_1(self):
		historical_time_periods = {0: [1, 3], 1: [3, 6],}
		expected = [11.0, 11.0, 13.0, 13.0, 13.0]
		historical_mi_array_kg_per_time_period = {
			0: np.array([10.0, 21.0]),
			1: np.array([22.0, 39.0]),
		}
		node = 1
		self.assert_mi_aggregation_unpacking(expected, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def assert_mi_aggregation_unpacking(self, expected, historical_time_periods, historical_mi_array_kg_per_time_period, node):
		historical_nitrate_days = []
		if len(historical_time_periods) > 0:
			for i in range(1, historical_time_periods[len(historical_time_periods) - 1][1]):
				historical_nitrate_days.append(date(2023, 1, i + 1))
		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historical_nitrate_days = historical_nitrate_days
		blackboard.historical_time_periods = historical_time_periods
		blackboard.historical_mi_array_kg_per_time_period = historical_mi_array_kg_per_time_period
		blackboard.node = node

		actual = historical_nitrate._calculate_aggregate_mi_unpacking(blackboard)

		np.testing.assert_allclose(expected, actual)
