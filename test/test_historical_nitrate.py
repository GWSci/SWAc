from datetime import date
import numpy as np
import swacmod.nitrate as nitrate
import unittest

class Test_Historical_Nitrate(unittest.TestCase):
	def test_historical_nitrate_returns_zeros_when_disabled(self):
		data = {
			"params": {
				"historical_nitrate_process": "disabled",
			}, "series" : {
				"date": [date(2023, 1, 1), date(2023, 1, 2), ]
			},
		}
		output = {}
		node = 3
		output = nitrate.get_historical_nitrate(data, output, node)
		actual = output["historical_nitrate_reaching_water_table_array_tons_per_day"]
		expected = np.zeros(2)
		np.testing.assert_array_almost_equal(expected, actual)
