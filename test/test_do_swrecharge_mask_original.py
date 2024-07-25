import unittest
import swacmod.model as m
import numpy as np

class Test_Do_Swrecharge_Mask_Original(unittest.TestCase):
	def test_Do_Swrecharge_Mask_Original_for_empty_model_leaves_runoff_and_recharge_unchanged(self):
		input_data, input_runoff, input_recharge = make_model(0)
		runoff, recharge = m.do_swrecharge_mask_original(input_data, input_runoff, input_recharge)
		self.assertEqual("some runoff", runoff)
		self.assertEqual("some recharge", recharge)

def make_model(node_count):
	input_data = {
		"params": {
			'num_nodes': node_count,
			'ror_prop': np.array([[]]),
			'ror_limit': np.array([[]]),
			'swrecharge_zone_mapping': {1: 1},
			'routing_topology': {}
		},
		"series": {
			'date': [],
			'months': []
		},
	}
	input_runoff = "some runoff"
	input_recharge = "some recharge"
	return input_data, input_runoff, input_recharge
