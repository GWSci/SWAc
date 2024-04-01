from datetime import date
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Surface_Nitrate_Aggregation(unittest.TestCase):
	def test_surface_nitrate_aggregation_for_empty_data(self):
		data = make_data(node_areas = {0: 10.0}, time_periods = {})
		output = {
			"nitrate_to_surface_water_array_tons_per_day" : np.array([]),
			"combined_str" : np.array([]),
		}
		node = 0

		actual = nitrate.make_aggregation_array(data)
		actual = nitrate.aggregate_surface_water_nitrate(actual, data, output, node)
		expected = np.zeros(shape = (0, 1))
		np.testing.assert_array_equal(expected, actual)

	def test_surface_nitrate_aggregation_for_one_day(self):
		data = make_data(node_areas = {0: 5}, time_periods = {0: [1, 2]})
		output = {
			"nitrate_to_surface_water_array_tons_per_day" : np.array([30.0]),
			"combined_str" : np.array([300.0]),
		}
		node = 0

		actual = nitrate.make_aggregation_array(data)
		actual = nitrate.aggregate_surface_water_nitrate(actual, data, output, node)
		expected = np.array([[15.0]])
		np.testing.assert_array_equal(expected, actual)

	def test_surface_nitrate_aggregation_for_two_nodes_and_one_day(self):
		data = make_data(node_areas = {0: 5, 1: 11}, time_periods = {0: [1, 2]})
		output_node_0 = {
			"nitrate_to_surface_water_array_tons_per_day" : np.array([30.0]),
			"combined_str" : np.array([300.0]),
		}
		node_0 = 0
		output_node_1 = {
			"nitrate_to_surface_water_array_tons_per_day" : np.array([1001.0]),
			"combined_str" : np.array([1300.0]),
		}
		node_1 = 1

		actual = nitrate.make_aggregation_array(data)
		actual = nitrate.aggregate_surface_water_nitrate(actual, data, output_node_0, node_0)
		actual = nitrate.aggregate_surface_water_nitrate(actual, data, output_node_1, node_1)
		expected = np.array([[15.0, 500.5]])
		np.testing.assert_array_equal(expected, actual)

	def test_surface_nitrate_aggregation_for_several_days_in_same_time_period(self):
		data = make_data(node_areas = {0: 5.0}, time_periods = {0: [1, 4]})
		output = {
			"nitrate_to_surface_water_array_tons_per_day" : np.array([4000.0, 400.0, 55.0]),
			"combined_str" : np.array([300.0, 1100.0, 1900.0]),
		}
		node = 0

		actual = nitrate.make_aggregation_array(data)
		actual = nitrate.aggregate_surface_water_nitrate(actual, data, output, node)
		expected = np.array([[1113.75]])
		np.testing.assert_array_equal(expected, actual)

	def test_surface_nitrate_aggregation_for_several_days_each_in_its_own_time_period(self):
		data = make_data(node_areas = {0: 5.0}, time_periods = {0: [1, 2], 1: [2, 3], 2: [3, 4]})
		output = {
			"nitrate_to_surface_water_array_tons_per_day" : np.array([435.0, 1705.0, 3515.0]),
			"combined_str" : np.array([300.0, 1100.0, 1900.0]),
		}
		node = 0

		actual = nitrate.make_aggregation_array(data)
		actual = nitrate.aggregate_surface_water_nitrate(actual, data, output, node)
		expected = np.array([[217.5], [852.5], [1757.5]])
		np.testing.assert_array_equal(expected, actual)

def make_data(node_areas, time_periods):
	dates = []
	if len(time_periods) > 0:
		for i in range(1, time_periods[len(time_periods) - 1][1]):
			dates.append(date(2023, 1, i + 1))
	return {
		"series": {
			"date" : dates
		}, "params" : {
			"node_areas" : node_areas,
			"time_periods" : time_periods
		}
	}
