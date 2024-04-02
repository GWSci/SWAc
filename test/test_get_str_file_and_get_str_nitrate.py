import unittest
import swacmod.model as m
import numpy as np
import warnings

class Test_Get_Str_File_And_Get_Str_Nitrate(unittest.TestCase):
	def test_get_str_file_for_1_node_and_1_sp(self):
		data = {
			"params": {
				"node_areas" : {1: 100.0},
				"run_name": "aardvark",
				"time_periods": {1: [1, 2]},
				"num_nodes": 1,
				"mf96_lrc": [1, 1, 1],
				"routing_topology": {1 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
				"istcb1": None,
				"istcb2": None,
			}
		}
		runoff = np.array([100.0, 200.0])

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			str = m.get_str_file(data, runoff)

		self.assertEqual(1, str.mxacts)
		self.assertEqual(1, str.nss)
		self.assertEqual(8, str.ntrib)
		self.assertEqual(0, str.ipakcb)
		self.assertIsNone(str.istcb2)
		np.testing.assert_array_almost_equal(
			np.array([[0.0, -2.0, 0.0, -2.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 111.111, 222.222]]),
			str.stress_period_data.get_dataframe())
		self.assertEqual({0: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, str.segment_data)
