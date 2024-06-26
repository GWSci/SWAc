import unittest

from swacmod import model as m
import numpy as np

class Test_Get_Flows(unittest.TestCase):
	def test_get_flows_for_an_empty_model_is_empty(self):
		sorted_by_ca = {}
		swac_seg_dic = None
		nodes = 0
		nss = 0
		source = None
		index_offset = None
		actual_A, actual_B = m.get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset)
		np.testing.assert_array_almost_equal([], actual_A)
		np.testing.assert_array_almost_equal([], actual_B)
