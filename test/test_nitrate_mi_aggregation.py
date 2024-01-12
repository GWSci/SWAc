from datetime import date
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Nitrate_mi_Aggregation(unittest.TestCase):
	def test_x(self):
		data = make_data(time_periods = {})
		output = {
			"mi_array_kg_per_day": np.array([])
		}
		node = 0

		actual = nitrate.make_mi_aggregation_array(data)
		actual = nitrate.aggregate_mi(actual, data, output, node)

		expected = np.array([])
		np.testing.assert_array_almost_equal(expected, actual)

def make_data(time_periods):
	dates = []
	if len(time_periods) > 0:
		for i in range(1, time_periods[len(time_periods) - 1][1]):
			dates.append(date(2023, 1, i + 1))
	return {
		"series": {
			"date" : dates
		}, "params" : {
			"time_periods" : time_periods
		}
	}
