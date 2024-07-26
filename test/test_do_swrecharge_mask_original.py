import unittest
import swacmod.model as m
import numpy as np
import datetime
import swacmod.finalization as finalization

class Test_Do_Swrecharge_Mask_Original(unittest.TestCase):
	def test_make_make_routing_topology(self):
		downstream_nodes = lambda node_count: [line[0] for line in make_routing_topology(node_count).values()]
		self.assertEqual([], downstream_nodes(0))
		self.assertEqual([-1], downstream_nodes(1))
		self.assertEqual([2, -1], downstream_nodes(2))
		self.assertEqual([2, 3, -1], downstream_nodes(3))
		self.assertEqual([2, 3, -1, -1], downstream_nodes(4))
		self.assertEqual([2, 3, -1, 5, -1], downstream_nodes(5))
		self.assertEqual([2, 3, 6, 5, 6, -1], downstream_nodes(6))
		self.assertEqual([2, 3, 6, 5, 6, -1, -1], downstream_nodes(7))
		self.assertEqual([2, 3, 6, 5, 6, -1, 8, -1], downstream_nodes(8))
		self.assertEqual([2, 3, 6, 5, 6, 9, 8, 9, -1], downstream_nodes(9))

	def test_Do_Swrecharge_Mask_Original_for_empty_model_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(0, 0)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1], runoff)
		self.assertEqual([-1], recharge)

	def test_Do_Swrecharge_Mask_Original_for_empty_model_and_1_day_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(0, 1)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1], runoff)
		self.assertEqual([-1], recharge)

	def test_Do_Swrecharge_Mask_Original_for_singleton_model_and_0_days_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(1, 0)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1], runoff)
		self.assertEqual([-1], recharge)

	def test_Do_Swrecharge_Mask_Original_for_singleton_model(self):
		input_data, input_runoff, input_recharge = make_model(1, 1)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1, 2], runoff)
		self.assertEqual([-1, 3], recharge)

	def test_Do_Swrecharge_Mask_Original_for_2_nodes_and_1_day_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(2, 1)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1, 1, 3], runoff)
		self.assertEqual([-1, 4, 7], recharge)

	def test_Do_Swrecharge_Mask_Original_for_9_nodes_and_1_day_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(9, 1)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], runoff)
		self.assertEqual([-1, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0], recharge)

	def test_Do_Swrecharge_Mask_Original_for_3_nodes_and_5_days_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(3, 5)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual([-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], runoff)
		self.assertEqual([-1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46], recharge)

def make_model(node_count, day_count):
	input_data = {
		"params": {
			'num_nodes': node_count,
			'ror_prop': np.array([[1.0], [0.0]] * 6),
			'ror_limit': np.array([[1.0], [0.0]] * 6),
			'swrecharge_zone_mapping': {(node_index + 1): 1 for node_index in range(node_count)},
			'routing_topology': make_routing_topology(node_count),
			'time_periods': [[1, day_count + 1]],
			'start_date': datetime.datetime(2024, 1, 1)
		},
		"series": {},
	}
	finalization.fin_date(input_data, "date")
	finalization.fin_months(input_data, "months")
	input_runoff = [-1]
	input_recharge = [-1]
	for i in range(node_count * day_count):
		input_runoff.append(2 * (i+1))
		input_recharge.append(3 * (i+1))
	return input_data, input_runoff, input_recharge

def make_routing_topology(node_count):
	"""
	Makes the pattern:
	1 -> 2 -> 3
	          V
	4 -> 5 -> 6
	          V
	7 -> 8 -> 9
	"""
	result = {}
	for node_index in range(node_count):
		node_number = node_index + 1
		if node_number % 3 == 0:
			downstream = node_number +  3
		else:
			downstream = node_number + 1
		if downstream > node_count:
			downstream = -1
		result[node_number] = [downstream, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	return result
