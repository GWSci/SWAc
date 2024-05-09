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

	def test_get_str_file_for_3_nodes_and_2_sp(self):
		data = {
			"params": {
				"node_areas" : {1: 2.0, 2: 3.0, 3: 5.0},
				"run_name": "aardvark",
				"time_periods": {1: [1, 2], 2: [2, 5]},
				"num_nodes": 3,
				"mf96_lrc": [1, 1, 3],
				"routing_topology": {1 : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3 : [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
				"istcb1": None,
				"istcb2": None,
			}
		}
		runoff = np.array([7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			str = m.get_str_file(data, runoff)

		self.assertEqual(3, str.mxacts)
		self.assertEqual(3, str.nss)
		self.assertEqual(8, str.ntrib)
		self.assertEqual(0, str.ipakcb)
		self.assertIsNone(str.istcb2)
		self.assertEqual(
			{
				0: [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
				1: [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
			},
			str.segment_data)

	def test_get_str_nitrate_for_3_nodes_and_2_sp(self):
		data = {
			"params": {
				"node_areas" : {1: 2.0, 2: 3.0, 3: 5.0},
				"run_name": "aardvark",
				"time_periods": {1: [1, 2], 2: [2, 5]},
				"num_nodes": 3,
				"mf96_lrc": [1, 1, 3],
				"routing_topology": {1 : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3 : [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
				"istcb1": None,
				"istcb2": None,
			}
		}
		runoff = np.array([7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])

		stream_nitrate_aggregation = np.array([[31.0, 37.0, 41.0], [43.0, 47.0, 53.0]])

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			actual = m.get_str_nitrate(data, runoff, stream_nitrate_aggregation)

		expected = [
			[0.0, 0.0, 482.352941],
			[0.0, 0.0, 365.517241]]
		np.testing.assert_array_almost_equal(expected, actual)

	def test_get_str_nitrate_for_4_nodes_3_str_and_2_sp(self):
		data = {
			"params": {
				"node_areas" : {1: 2.0, 2: 3.0, 3: 5.0, 4: 1.0},
				"run_name": "aardvark",
				"time_periods": {1: [1, 2], 2: [2, 5]},
				"num_nodes": 4,
				"mf96_lrc": [1, 1, 4],
				"routing_topology": {1 : [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3 : [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
				"istcb1": None,
				"istcb2": None,
			}
		}
		runoff = np.array([7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0])

		stream_nitrate_aggregation = np.array([[31.0, 37.0, 41.0], [43.0, 47.0, 53.0]])

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			actual = m.get_str_nitrate(data, runoff, stream_nitrate_aggregation)

		expected = [
			[0.0, 0.0, 482.352941],
			[0.0, 0.0, 341.935484]]
		np.testing.assert_array_almost_equal(expected, actual)
