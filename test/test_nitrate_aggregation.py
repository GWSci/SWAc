import numpy as np
import unittest

class Test_Nitrate_Aggregation(unittest.TestCase):
	def test_x(self):
		data = {
			"params" : {
				"time_periods" : {}
			}
		}
		output = None
		node = None

		actual = aggregate_nitrate(data, output, node)
		expected = np.zeros(shape = (0, 0))
		np.testing.assert_array_equal(expected, actual)

def aggregate_nitrate(data, output, node):
	return np.zeros(shape = (0, 0))