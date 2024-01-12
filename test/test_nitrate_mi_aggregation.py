from datetime import date
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Nitrate_mi_Aggregation(unittest.TestCase):
	def test_nitrate_mi_aggregation_for_empty_data(self):
		data = make_data(node_count = 1, time_periods = {})
		output = {
			"mi_array_kg_per_day": np.array([[]])
		}
		node = 0
		output_for_nodes = [output]

		expected = np.zeros(shape = (1, 0))

		actual = nitrate.make_mi_aggregation_array(data)
		for node, output in enumerate(output_for_nodes):
			actual = nitrate.aggregate_mi(actual, data, output, node)

		np.testing.assert_allclose(expected, actual)

	def test_nitrate_mi_aggregation_for_one_day(self):
		data = make_data(node_count = 1, time_periods = {0: [1, 2]})
		output = {
			"mi_array_kg_per_day": np.array([3.0])
		}
		node = 0
		output_for_nodes = [output]

		expected = np.array([[3.0]])

		actual = nitrate.make_mi_aggregation_array(data)
		for node, output in enumerate(output_for_nodes):
			actual = nitrate.aggregate_mi(actual, data, output, node)

		np.testing.assert_allclose(expected, actual)

	def test_nitrate_mi_aggregation_for_two_nodes_and_one_day(self):
		data = make_data(node_count = 2, time_periods = {0: [1, 2]})
		output_0 = {
			"mi_array_kg_per_day": np.array([3.0])
		}
		output_1 = {
			"mi_array_kg_per_day": np.array([5.0])
		}
		output_for_nodes = [output_0, output_1]

		expected = np.array([[3.0], [5.0]])

		actual = nitrate.make_mi_aggregation_array(data)
		for node, output in enumerate(output_for_nodes):
			actual = nitrate.aggregate_mi(actual, data, output, node)

		np.testing.assert_allclose(expected, actual)

def make_data(node_count, time_periods):
	dates = []
	if len(time_periods) > 0:
		for i in range(1, time_periods[len(time_periods) - 1][1]):
			dates.append(date(2023, 1, i + 1))
	return {
		"series": {
			"date" : dates
		}, "params" : {
			"time_periods" : time_periods,
			"num_nodes" : node_count,
		}
	}
