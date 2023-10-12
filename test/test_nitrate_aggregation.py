from datetime import date
import numpy as np
import unittest

class Test_Nitrate_Aggregation(unittest.TestCase):
	def test_nitrate_aggregation_for_empty_data(self):
		data = make_data(node_areas = {0: [10.0]}, time_periods = {})
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([]),
			"combined_recharge" : np.array([]),
		}
		node = 0

		actual = make_aggregation_array(data)
		actual = aggregate_nitrate(actual, data, output, node)
		expected = np.zeros(shape = (0, 1))
		np.testing.assert_array_equal(expected, actual)

	def test_nitrate_aggregation_for_one_day(self):
		data = make_data(node_areas = {0: [5]}, time_periods = {0: [1, 2]})
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([30]),
			"combined_recharge" : np.array([300.0]),
		}
		node = 0

		actual = make_aggregation_array(data)
		actual = aggregate_nitrate(actual, data, output, node)
		expected = np.array([[2.0]])
		np.testing.assert_array_equal(expected, actual)

	def test_nitrate_aggregation_for_two_nodes_and_one_day(self):
		data = make_data(node_areas = {0: [5], 1: [11]}, time_periods = {0: [1, 2]})
		output_node_0 = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([30]),
			"combined_recharge" : np.array([300.0]),
		}
		node_0 = 0
		output_node_1 = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([1001]),
			"combined_recharge" : np.array([1300.0]),
		}
		node_1 = 1

		actual = make_aggregation_array(data)
		actual = aggregate_nitrate(actual, data, output_node_0, node_0)
		actual = aggregate_nitrate(actual, data, output_node_1, node_1)
		expected = np.array([[2.0, 7.0]])
		np.testing.assert_array_equal(expected, actual)

	def test_nitrate_aggregation_for_several_days_in_same_time_period(self):
		data = make_data(node_areas = {0: [5.0]}, time_periods = {0: [1, 4]})
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([4000.0, 400.0, 55.0]),
			"combined_recharge" : np.array([300.0, 1100.0, 1900.0]),
		}
		node = 0

		actual = make_aggregation_array(data)
		actual = aggregate_nitrate(actual, data, output, node)
		expected = np.array([[27.0]])
		np.testing.assert_array_equal(expected, actual)

	def test_nitrate_aggregation_for_several_days_each_in_its_own_time_period(self):
		data = make_data(node_areas = {0: [5.0]}, time_periods = {0: [1, 2], 1: [2, 3], 2: [3, 4]})
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([435.0, 1705.0, 3515.0]),
			"combined_recharge" : np.array([300.0, 1100.0, 1900.0]),
		}
		node = 0

		actual = make_aggregation_array(data)
		actual = aggregate_nitrate(actual, data, output, node)
		expected = np.array([[29.0], [31.0], [37.0]])
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

def make_aggregation_array(data):
	time_periods = data["params"]["time_periods"]
	node_areas = data["params"]["node_areas"]
	shape = (len(time_periods), len(node_areas))
	aggregation = np.zeros(shape = shape)
	return aggregation

def aggregate_nitrate(aggregation, data, output, node):
	time_periods = data["params"]["time_periods"]
	nitrate_reaching_water_table_array_tons_per_day = output["nitrate_reaching_water_table_array_tons_per_day"]
	combined_recharge_m_cubed = _calculate_combined_recharge_m_cubed(data, output, node)

	for time_period_index in range(len(time_periods)):
		time_period = time_periods[time_period_index]
		first_day_index = time_period[0] - 1
		last_day_index = time_period[1] - 1
		sum_of_recharge_m_cubed = 0.0
		for day_index in range(first_day_index, last_day_index):
			sum_of_recharge_m_cubed += combined_recharge_m_cubed[day_index]
		sum_of_nitrate_tons = nitrate_reaching_water_table_array_tons_per_day[first_day_index:last_day_index].sum()
		sum_of_recharge_m_cubed = combined_recharge_m_cubed[first_day_index:last_day_index].sum()
		aggregation[time_period_index, node] += sum_of_nitrate_tons / sum_of_recharge_m_cubed

	return aggregation

def _calculate_combined_recharge_m_cubed(data, output, node):
	node_areas = data["params"]["node_areas"]
	combined_recharge_mm = output["combined_recharge"]
	combined_recharge_m = _convert_mm_to_m(combined_recharge_mm)
	combined_recharge_m_cubed = combined_recharge_m * node_areas[node][0]
	return combined_recharge_m_cubed

def _convert_mm_to_m(arr):
	return arr / 100.0