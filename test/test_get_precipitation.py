import numpy
import swacmod.model
import unittest

# Unless otherwise stated, the tests assume 3 nodes and 5 times

rainfall_ts = numpy.array([
	[0, 1, 2, 3, 4, 5],
	[0, 2, 4, 6, 8, 10],
	[0, 3, 6, 9, 12, 15],
	[0, 4, 8, 12, 16, 20],
	[0, 5, 10, 15, 20, 25],
], dtype=float)

rainfall_zone_mapping = [
	[2, 2, 5,],
	[3, 7, 13,],
	[5, 17, 23,],
]

expected_precipitation = numpy.array([
	[2*1, 2*2, 2*3, 2*4, 2*5],
	[7*2, 7*4, 7*6, 7*8, 7*10],
	[17*4, 17*8, 17*12, 17*16, 17*20],
], dtype=float)

class Test_Get_Precipitation(unittest.TestCase):
	def test_get_precipitation(self):
		input_data = {
			"series": {
				"rainfall_ts": rainfall_ts
			},
			"params": {
				"rainfall_zone_mapping": rainfall_zone_mapping
			},
		}
		actual_oracle = oracle_get_precipitation(input_data)
		actual_numpy = numpy_get_precipitation(input_data)
		numpy.testing.assert_array_equal(expected_precipitation, actual_oracle)
		numpy.testing.assert_array_equal(expected_precipitation, actual_numpy)

def oracle_get_precipitation(data):
	result = []
	for node in range(0, 3):
		precipitation = swacmod.model.get_precipitation(data, {}, node)
		result.append(precipitation["rainfall_ts"])
	return numpy.array(result)

def numpy_get_precipitation(data):
	series, params = data['series'], data['params']
	rainfall_zone_mapping = params['rainfall_zone_mapping']
	zone_rf = numpy.array(list(map(lambda x: x[0] - 1, rainfall_zone_mapping)))
	coef_rf = numpy.array(list(map(lambda x: x[1], rainfall_zone_mapping)))
	rainfall_ts = numpy.transpose(series['rainfall_ts'][:, zone_rf]) * coef_rf[:, None]
	return rainfall_ts
