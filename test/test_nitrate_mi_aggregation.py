from datetime import date
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Nitrate_mi_Aggregation(unittest.TestCase):
	def test_nitrate_mi_aggregation_for_empty_data(self):
		node_count = 1
		time_periods = {}
		output_per_node = [[]]
		expected = np.zeros(shape = (1, 0))
		self.assert_mi_aggregation(expected, node_count, time_periods, output_per_node)

	def test_nitrate_mi_aggregation_for_one_day(self):
		node_count = 1
		time_periods = {0: [1, 2]}
		output_per_node = [[3.0]]
		expected = np.array([[3.0]])
		self.assert_mi_aggregation(expected, node_count, time_periods, output_per_node)

	def test_nitrate_mi_aggregation_for_two_nodes_and_one_day(self):
		node_count = 2
		time_periods = {0: [1, 2]}
		output_per_node = [[3.0], [5.0]]
		expected = np.array([[3.0], [5.0]])
		self.assert_mi_aggregation(expected, node_count, time_periods, output_per_node)

	def test_nitrate_mi_aggregation_for_several_days_in_same_time_period(self):
		node_count = 1
		time_periods = {0: [1, 4]}
		output_per_node = [[1.0, 20.0, 300.0]]
		expected = np.array([[321.0]])
		self.assert_mi_aggregation(expected, node_count, time_periods, output_per_node)

	def test_nitrate_mi_aggregation_for_several_days_each_in_its_own_time_period(self):
		node_count = 1
		time_periods = {0: [1, 2], 1: [2, 3], 2: [3, 4]}
		output_per_node = [[1.0, 20.0, 300.0]]
		expected = np.array([[1.0, 20.0, 300.0]])
		self.assert_mi_aggregation(expected, node_count, time_periods, output_per_node)

	def assert_mi_aggregation(self, expected, node_count, time_periods, output_per_node):
		dates = []
		if len(time_periods) > 0:
			for i in range(1, time_periods[len(time_periods) - 1][1]):
				dates.append(date(2023, 1, i + 1))
		data = {
			"series": {
				"date" : dates
			}, "params" : {
				"time_periods" : time_periods,
				"num_nodes" : node_count,
			}
		}

		actual = nitrate.make_mi_aggregation_array(data)
		for node in range(len(output_per_node)):
			output = {
				"mi_array_kg_per_day": np.array(output_per_node[node])
			}
			actual = nitrate.aggregate_mi(actual, data, output, node)

		np.testing.assert_allclose(expected, actual)
