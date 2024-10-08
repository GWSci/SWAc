import numpy
import swacmod.model_numpy
import unittest

# Unless otherwise stated, the tests assume 3 nodes and 5 times

rainfall_ts = numpy.array([
	[0, 1, 2, 3, 4, 5],
	[0, 2, 4, 6, 8, 10],
	[0, 3, 6, 9, 12, 15],
	[0, 4, 8, 12, 16, 20],
	[0, 5, 10, 15, 20, 25],
], dtype=float)

rainfall_zone_mapping = {
	1: [2, 2,],
	2: [3, 7,],
	3: [5, 17,],
}

expected_precipitation = numpy.array([
	[2*1, 2*2, 2*3, 2*4, 2*5],
	[7*2, 7*4, 7*6, 7*8, 7*10],
	[17*4, 17*8, 17*12, 17*16, 17*20],
], dtype=float)

nodes = range(1, 4)

class Test_Get_Precipitation(unittest.TestCase):
	def test_get_precipitation_oracle(self):
		input_data = {
			"series": {
				"rainfall_ts": rainfall_ts
			},
			"params": {
				"rainfall_zone_mapping": rainfall_zone_mapping
			},
		}
		actual_oracle = oracle_get_precipitation(input_data)
		numpy.testing.assert_array_equal(expected_precipitation, actual_oracle)

def oracle_get_precipitation(data):
	result = []
	for node in nodes:
		precipitation = get_precipitation(data, {}, node)
		result.append(precipitation["rainfall_ts"])
	return numpy.array(result)

def get_precipitation(data, output, node):
    """C) Precipitation [mm/d]."""
    series, params = data['series'], data['params']
    zone_rf = params['rainfall_zone_mapping'][node][0] - 1
    coef_rf = params['rainfall_zone_mapping'][node][1]
    rainfall_ts = series['rainfall_ts'][:, zone_rf] * coef_rf
    return {'rainfall_ts': rainfall_ts}
