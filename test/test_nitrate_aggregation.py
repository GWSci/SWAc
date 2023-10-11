import numpy as np
import unittest

class Test_Nitrate_Aggregation(unittest.TestCase):
	def test_nitrate_aggregation_for_empty_data(self):
		data = {
			"params" : {
				"node_areas" : {},
				"time_periods" : {}
			}
		}
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([]),
			"combined_recharge" : np.array([]),
		}
		node = None

		actual = aggregate_nitrate(data, output, node)
		expected = np.zeros(shape = (0, 0))
		np.testing.assert_array_equal(expected, actual)

	def test_nitrate_aggregation_for_one_day(self):
		data = {
			"params" : {
				"node_areas" : {
					0: [5]
				}, "time_periods" : {
					0: [1, 2]
				}
			}
		}
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([30]),
			"combined_recharge" : np.array([300.0]),
		}
		node = 0

		actual = aggregate_nitrate(data, output, node)
		expected = np.array([[2.0]])
		np.testing.assert_array_equal(expected, actual)

def aggregate_nitrate(data, output, node):
	time_periods = data["params"]["time_periods"]
	node_areas = data["params"]["node_areas"]
	num_nodes = len(node_areas)
	nitrate_reaching_water_table_array_tons_per_day = output["nitrate_reaching_water_table_array_tons_per_day"]
	combined_recharge_mm = output["combined_recharge"]
	shape = (len(time_periods), num_nodes)
	result = np.zeros(shape = shape)
	if (num_nodes > 0):
		combined_recharge_m = _convert_mm_to_m(combined_recharge_mm)
		result[0, 0] = nitrate_reaching_water_table_array_tons_per_day[0] / (combined_recharge_m * node_areas[node][0])
	return result

def _convert_mm_to_m(arr):
	return arr / 100.0