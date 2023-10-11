from datetime import date
import numpy as np
import unittest

class Test_Nitrate_Aggregation(unittest.TestCase):
	def test_nitrate_aggregation_for_empty_data(self):
		data = {
			"series": {
				"date" : [date(2023, 9, 28)]
			}, "params" : {
				"node_areas" : {
					0: [10.0]
				},
				"time_periods" : {}
			}
		}
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([]),
			"combined_recharge" : np.array([]),
		}
		node = 0

		actual = aggregate_nitrate(data, output, node)
		expected = np.zeros(shape = (0, 1))
		np.testing.assert_array_equal(expected, actual)

	def test_nitrate_aggregation_for_one_day(self):
		data = {
			"series": {
				"date" : [date(2023, 9, 28)]
			}, "params" : {
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

	def test_nitrate_aggregation_for_several_days_in_same_time_period(self):
		data = {
			"series": {
				"date" : [date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)]
			}, "params" : {
				"node_areas" : {
					0: [5.0, 13.0, 23.0]
				}, "time_periods" : {
					0: [1, 4]
				}
			}
		}
		output = {
			"nitrate_reaching_water_table_array_tons_per_day" : np.array([10000.0, 6000.0, 65.0]),
			"combined_recharge" : np.array([300.0, 1100.0, 1900.0]),
		}
		node = 0

		actual = aggregate_nitrate(data, output, node)
		expected = np.array([[27.0]])
		np.testing.assert_array_equal(expected, actual)

def aggregate_nitrate(data, output, node):
	dates = data["series"]["date"]
	time_periods = data["params"]["time_periods"]
	node_areas = data["params"]["node_areas"]
	nitrate_reaching_water_table_array_tons_per_day = output["nitrate_reaching_water_table_array_tons_per_day"]
	combined_recharge_mm = output["combined_recharge"]
	combined_recharge_m = _convert_mm_to_m(combined_recharge_mm)
	combined_recharge_m_cubed = combined_recharge_m * node_areas[node]
	shape = (len(time_periods), len(node_areas))
	result = np.zeros(shape = shape)
	if (len(node_areas) > 0) and (len(time_periods) > 0):
		sum_of_nitrate_tons = 0.0
		sum_of_recharge_m_cubed = 0.0
		for day in range(len(dates)):
			sum_of_nitrate_tons += nitrate_reaching_water_table_array_tons_per_day[day]
			sum_of_recharge_m_cubed += combined_recharge_m_cubed[day]
		result[0, 0] += sum_of_nitrate_tons / sum_of_recharge_m_cubed
	return result

def _convert_mm_to_m(arr):
	return arr / 100.0