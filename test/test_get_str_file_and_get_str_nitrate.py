import unittest
import swacmod.model as m
import numpy as np
import warnings
import swacmod.feature_flags as ff

class Test_Get_Str_File_And_Get_Str_Nitrate(unittest.TestCase):
	def test_get_str_file_for_3_nodes_and_1_sp(self):
		data = {
			"params" : {
				"node_areas" : [-1, 100, 200, 300],
				"run_name" : "str-aardvark",
				"time_periods" : [1, 2],
				"num_nodes" : 3,
				"mf96_lrc" : [1, 1, 3],
				"routing_topology" : {
					1 : [1, 1, 1, 40, 50, 60, 70, 80, 90, 100],
					2 : [2, 1, 2, 40, 50, 60, 70, 80, 90, 100],
					3 : [3, 1, 3, 40, 50, 60, 70, 80, 90, 100],
				},
				"istcb1" : 0,
				"istcb2" : 0,
			}
		}
		runoff = [-1, 5, 7, 11, 13, 17, 19]

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=UserWarning)
			str = m.get_str_file(data, runoff)

		self.assertEqual(3, str.mxacts)
		self.assertEqual(3, str.nss)
		self.assertEqual(8, str.ntrib)
		self.assertEqual(0, str.ipakcb)
		self.assertEqual(0, str.istcb2)

		actual_sp_data = str.stress_period_data.get_dataframe()

		np.testing.assert_array_equal([0, 1, 2, 3, 4, 5], actual_sp_data.index.values)
		self.assertListEqual(
			["per", "k", "i", "j", "node", "segment", "reach", "flow", "stage", "cond", "sbot", "stop", "width", "slope", "rough"],
			list(actual_sp_data.columns.values))

		self.assertEqual(0, actual_sp_data.at[0, "per"])
		self.assertEqual(-1, actual_sp_data.at[0, "k"])
		self.assertEqual(-1, actual_sp_data.at[0, "i"])
		self.assertEqual(0, actual_sp_data.at[0, "j"])
		self.assertEqual(-3, actual_sp_data.at[0, "node"])
		self.assertEqual(1, actual_sp_data.at[0, "segment"])
		self.assertEqual(1, actual_sp_data.at[0, "reach"])
		self.assertEqual(0.0, actual_sp_data.at[0, "flow"])
		# self.assertEqual(0.0, actual_sp_data.at[0, "flow1"])
		self.assertEqual(150.0, actual_sp_data.at[0, "stage"])
		self.assertAlmostEqual(4571.4287, actual_sp_data.at[0, "cond"], places = 4)
		self.assertEqual(-10.0, actual_sp_data.at[0, "sbot"])
		self.assertEqual(60.0, actual_sp_data.at[0, "stop"])
		self.assertEqual(100.0, actual_sp_data.at[0, "width"])
		self.assertAlmostEqual(111.111, actual_sp_data.at[0, "slope"], places = 4)
		self.assertAlmostEqual(222.222, actual_sp_data.at[0, "rough"], places = 4)

		self.assertEqual(0, actual_sp_data.at[1, "per"])
		self.assertEqual(-1, actual_sp_data.at[1, "k"])
		self.assertEqual(-1, actual_sp_data.at[1, "i"])
		self.assertEqual(1, actual_sp_data.at[1, "j"])
		self.assertEqual(-2, actual_sp_data.at[1, "node"])
		self.assertEqual(2, actual_sp_data.at[1, "segment"])
		self.assertEqual(1, actual_sp_data.at[1, "reach"])
		self.assertAlmostEqual(1.4, actual_sp_data.at[1, "flow"], places = 4)
		# self.assertAlmostEqual(3.4, actual_sp_data.at[1, "flow1"], places = 4)
		self.assertEqual(150.0, actual_sp_data.at[1, "stage"])
		self.assertAlmostEqual(4571.4287, actual_sp_data.at[1, "cond"], places = 4)
		self.assertEqual(-10.0, actual_sp_data.at[1, "sbot"])
		self.assertEqual(60.0, actual_sp_data.at[1, "stop"])
		self.assertEqual(100.0, actual_sp_data.at[1, "width"])
		self.assertAlmostEqual(111.111, actual_sp_data.at[1, "slope"], places = 4)
		self.assertAlmostEqual(222.222, actual_sp_data.at[1, "rough"], places = 4)

		self.assertEqual(0, actual_sp_data.at[2, "per"])
		self.assertEqual(-1, actual_sp_data.at[2, "k"])
		self.assertEqual(-1, actual_sp_data.at[2, "i"])
		self.assertEqual(2, actual_sp_data.at[2, "j"])
		self.assertEqual(-1, actual_sp_data.at[2, "node"])
		self.assertEqual(3, actual_sp_data.at[2, "segment"])
		self.assertEqual(1, actual_sp_data.at[2, "reach"])
		self.assertAlmostEqual(3.3, actual_sp_data.at[2, "flow"], places = 4)
		# self.assertAlmostEqual(5.7, actual_sp_data.at[2, "flow1"], places = 4)
		self.assertEqual(150.0, actual_sp_data.at[2, "stage"])
		self.assertAlmostEqual(4571.4287, actual_sp_data.at[2, "cond"], places = 4)
		self.assertEqual(-10.0, actual_sp_data.at[2, "sbot"])
		self.assertEqual(60.0, actual_sp_data.at[2, "stop"])
		self.assertEqual(100.0, actual_sp_data.at[2, "width"])
		self.assertAlmostEqual(111.111, actual_sp_data.at[2, "slope"], places = 4)
		self.assertAlmostEqual(222.222, actual_sp_data.at[2, "rough"], places = 4)

		self.assertEqual(1, actual_sp_data.at[3, "per"])
		self.assertEqual(-1, actual_sp_data.at[3, "k"])
		self.assertEqual(-1, actual_sp_data.at[3, "i"])
		self.assertEqual(0, actual_sp_data.at[3, "j"])
		self.assertEqual(-3, actual_sp_data.at[3, "node"])
		self.assertEqual(1, actual_sp_data.at[3, "segment"])
		self.assertEqual(1, actual_sp_data.at[3, "reach"])
		self.assertEqual(0.0, actual_sp_data.at[3, "flow"])
		self.assertEqual(150.0, actual_sp_data.at[3, "stage"])
		self.assertAlmostEqual(4571.4287, actual_sp_data.at[3, "cond"], places = 4)
		self.assertEqual(-10.0, actual_sp_data.at[3, "sbot"])
		self.assertEqual(60.0, actual_sp_data.at[3, "stop"])
		self.assertEqual(100.0, actual_sp_data.at[3, "width"])
		self.assertAlmostEqual(111.111, actual_sp_data.at[3, "slope"], places = 4)
		self.assertAlmostEqual(222.222, actual_sp_data.at[3, "rough"], places = 4)

		self.assertEqual(1, actual_sp_data.at[4, "per"])
		self.assertEqual(-1, actual_sp_data.at[4, "k"])
		self.assertEqual(-1, actual_sp_data.at[4, "i"])
		self.assertEqual(1, actual_sp_data.at[4, "j"])
		self.assertEqual(-2, actual_sp_data.at[4, "node"])
		self.assertEqual(2, actual_sp_data.at[4, "segment"])
		self.assertEqual(1, actual_sp_data.at[4, "reach"])
		self.assertAlmostEqual(3.4, actual_sp_data.at[4, "flow"], places = 4)
		self.assertEqual(150.0, actual_sp_data.at[4, "stage"])
		self.assertAlmostEqual(4571.4287, actual_sp_data.at[4, "cond"], places = 4)
		self.assertEqual(-10.0, actual_sp_data.at[4, "sbot"])
		self.assertEqual(60.0, actual_sp_data.at[4, "stop"])
		self.assertEqual(100.0, actual_sp_data.at[4, "width"])
		self.assertAlmostEqual(111.111, actual_sp_data.at[4, "slope"], places = 4)
		self.assertAlmostEqual(222.222, actual_sp_data.at[4, "rough"], places = 4)

		self.assertEqual(1, actual_sp_data.at[5, "per"])
		self.assertEqual(-1, actual_sp_data.at[5, "k"])
		self.assertEqual(-1, actual_sp_data.at[5, "i"])
		self.assertEqual(2, actual_sp_data.at[5, "j"])
		self.assertEqual(-1, actual_sp_data.at[5, "node"])
		self.assertEqual(3, actual_sp_data.at[5, "segment"])
		self.assertEqual(1, actual_sp_data.at[5, "reach"])
		self.assertAlmostEqual(5.7, actual_sp_data.at[5, "flow"], places = 4)
		self.assertEqual(150.0, actual_sp_data.at[5, "stage"])
		self.assertAlmostEqual(4571.4287, actual_sp_data.at[5, "cond"], places = 4)
		self.assertEqual(-10.0, actual_sp_data.at[5, "sbot"])
		self.assertEqual(60.0, actual_sp_data.at[5, "stop"])
		self.assertEqual(100.0, actual_sp_data.at[5, "width"])
		self.assertAlmostEqual(111.111, actual_sp_data.at[5, "slope"], places = 4)
		self.assertAlmostEqual(222.222, actual_sp_data.at[5, "rough"], places = 4)

		self.assertEqual({
			0: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
			1: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
		}, str.segment_data)

	def test_get_str_file_for_3_nodes_and_2_sp_when_use_natproc_is_true(self):
		original_use_natproc = ff.use_natproc
		ff.use_natproc = True
		try:
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
				warnings.filterwarnings("ignore", category=UserWarning)
				str = m.get_str_file(data, runoff)

			self.assertEqual(3, str.mxacts)
			self.assertEqual(3, str.nss)
			self.assertEqual(8, str.ntrib)
			self.assertEqual(0, str.ipakcb)
			self.assertIsNone(str.istcb2)

			expected_segment_data = {
				0: [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
				1: [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
			}
			self.assertEqual(expected_segment_data, str.segment_data)
		finally:
			ff.use_natproc = original_use_natproc

	def test_get_str_file_for_3_nodes_and_2_sp_when_use_natproc_is_false(self):
		original_use_natproc = ff.use_natproc
		ff.use_natproc = False
		try:
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
				warnings.filterwarnings("ignore", category=UserWarning)
				str = m.get_str_file(data, runoff)

			self.assertEqual(3, str.mxacts)
			self.assertEqual(3, str.nss)
			self.assertEqual(8, str.ntrib)
			self.assertEqual(0, str.ipakcb)
			self.assertIsNone(str.istcb2)

			expected_segment_data = {
				0: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
				1: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
			}

			self.assertEqual(expected_segment_data, str.segment_data)
		finally:
			ff.use_natproc = original_use_natproc

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

		actual = m.get_str_nitrate(data, runoff, stream_nitrate_aggregation)

		expected = [
			[0.0, 0.0, 482.352941],
			[0.0, 0.0, 341.935484]]
		np.testing.assert_array_almost_equal(expected, actual)
