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

	def test_get_flows_for_a_model_size_1_and_no_stream_cells_is_empty(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0)}
		swac_seg_dic = None
		nodes = 1
		nss = 0
		source = None
		index_offset = None
		actual_A, actual_B = m.get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset)
		np.testing.assert_array_almost_equal([], actual_A)
		np.testing.assert_array_almost_equal([], actual_B)

def make_routing_parameters(
	downstr = -1, # swac node downstream of this one
	str_flag = 0, # 1 = stream cell, 0 = *not* stream cell
	node_mf = -1, # modflow node number of this node 
	RCHLEN = -1, # the modflow SFR variable RCHLEN
	ca = -1, # the upstream contributing area 
	STRTOP = -1, # the modflow SFR variable STRTOP
	STRTHICK = -1, # the modflow SFR variable STRTHICK
	STRHC1 = -1, # the modflow SFR variable STRHC1
	DEPTH2 = -1, # the modflow SFR variable DEPTH2
	WIDTH2 = -1, # the modflow SFR variable WIDTH2
):
	return [downstr, str_flag, node_mf, RCHLEN, ca, STRTOP, STRTHICK, STRHC1, DEPTH2, WIDTH2]