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
