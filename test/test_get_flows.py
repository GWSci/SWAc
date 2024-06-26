import unittest

from swacmod import model as m
import numpy as np

class Test_Get_Flows(unittest.TestCase):
	def test_get_flows_for_an_empty_model_is_empty(self):
		sorted_by_ca = {}
		swac_seg_dic = None
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, swac_seg_dic)
		np.testing.assert_array_almost_equal([], actual_A)
		np.testing.assert_array_almost_equal([], actual_B)

	def test_get_flows_for_a_model_size_1_and_no_stream_cells_is_empty(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0)}
		swac_seg_dic = None
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, swac_seg_dic)
		np.testing.assert_array_almost_equal([], actual_A)
		np.testing.assert_array_almost_equal([], actual_B)

	def test_get_flows_for_a_model_size_1_and_1_stream_cell_has_zero_results(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0, str_flag = 1)}
		swac_seg_dic = None
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, swac_seg_dic)
		np.testing.assert_array_almost_equal([0], actual_A)
		np.testing.assert_array_almost_equal([0], actual_B)

	def test_get_flows_for_a_model_size_2_and_no_stream_cells_is_empty(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 0),
			2 : make_routing_parameters(downstr = 0),
		}
		swac_seg_dic = None
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, swac_seg_dic)
		np.testing.assert_array_almost_equal([], actual_A)
		np.testing.assert_array_almost_equal([], actual_B)

	def test_get_flows_for_a_model_size_2_and_no_stream_cells_and_downstream_connections_is_empty(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2),
			2 : make_routing_parameters(downstr = 0),
		}
		swac_seg_dic = None
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, swac_seg_dic)
		np.testing.assert_array_almost_equal([], actual_A)
		np.testing.assert_array_almost_equal([], actual_B)

	def test_get_flows_for_a_model_size_2_and_2_stream_cells_and_downstream_connections_is_empty(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 1, str_flag = 1),
		}
		swac_seg_dic = {}
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, swac_seg_dic)
		np.testing.assert_array_almost_equal([0, 2], actual_A)
		np.testing.assert_array_almost_equal([0, 0], actual_B)

def get_flows_adaptor(sorted_by_ca, swac_seg_dic):
	swac_seg_dic = {}
	stream_index = 0
	for node_number, params in sorted_by_ca.items():
		if (params[1] == 1):
			swac_seg_dic[node_number] = stream_index
			stream_index += 1

	nodes = len(sorted_by_ca)
	nss = len(list(filter(lambda x : x[1] == 1, sorted_by_ca.values())))
	long_list_for_source = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
	source = [-1000, -2000, -3000] + long_list_for_source[:nodes]
	index_offset = 3
	actual_A, actual_B = m.get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset)
	return actual_A, actual_B

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