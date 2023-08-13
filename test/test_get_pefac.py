import unittest
import numpy as np
import swacmod.model_plain_python as model
import swacmod.model_numpy as model_numpy
import swacmod.timer as timer

class test_get_pefac(unittest.TestCase):
	def test_x(self):
		data = {
			"time_switcher": timer.make_time_switcher(),
			"series": {
				"date": [0, 0, 0, 0, 0], # Doesn't seem to matter what this is. Only its length is used.
				"months": np.array([0, 2, 1, 4, 3], dtype=np.int64),
			},
			"params": {
				"kc_list": np.array([
					[ 2,  3,  5],
					[ 7, 11, 13],
					[17, 19, 23],
					[29, 31, 37],
					[41, 43, 47],
				], dtype=np.float64),
				"lu_spatial": np.array([
					[53, 59, 61],
					[67, 71, 73],
				], dtype=np.float64),
				"fao_process": "enabled",
				"canopy_process": "enabled",
			},
		}
		output = {
			"pe_ts": np.array([79, 83, 97, 101, 103], dtype=np.float64),
		}
		node = 1
		expected = model.get_pefac(data, output, node)["pefac"]
		actual = model_numpy.get_pefac(data, output, node)["pefac"]
		np.testing.assert_array_equal(expected, actual)
