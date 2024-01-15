from datetime import date
import swacmod.historical_nitrate as historical_nitrate
import unittest
import numpy as np

class Test_Nitrate_mi_Aggregation_Unpacking(unittest.TestCase):
	def test_nitrate_mi_aggregation_unpacking_for_one_day(self):
		node_count = 1
		historical_time_periods = {0: [1, 2]}
		expected = [3.0]
		historical_mi_array_kg_per_time_period = np.array({0: [3.0]})
		node = 0
		self.assert_mi_aggregation_unpacking(expected, node_count, historical_time_periods, historical_mi_array_kg_per_time_period, node)

	def assert_mi_aggregation_unpacking(self, expected, node_count, historical_time_periods, historical_mi_array_kg_per_time_period, node):
		historical_nitrate_days = []
		if len(historical_time_periods) > 0:
			for i in range(1, historical_time_periods[len(historical_time_periods) - 1][1]):
				historical_nitrate_days.append(date(2023, 1, i + 1))
		data = {
			"series": {	
			}, "params" : {
				"num_nodes" : node_count,
				"historical_nitrate_mi_array_kg_per_time_period" : historical_mi_array_kg_per_time_period,
			}
		}

		blackboard = historical_nitrate.HistoricalNitrateBlackboard()
		blackboard.historical_nitrate_days = historical_nitrate_days
		blackboard.historical_time_periods = historical_time_periods
		blackboard.historical_mi_array_kg_per_time_period = historical_mi_array_kg_per_time_period
		blackboard.node = node

		blackboard = historical_nitrate._calculate_aggregate_mi_unpacking(blackboard)
		actual = blackboard.historical_mi_array_kg_per_day

		np.testing.assert_allclose(expected, actual)
