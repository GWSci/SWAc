import unittest
import numpy as np

class Test_Get_Attenuated_Sfr_Flows(unittest.TestCase):
	def test_get_flows_for_an_empty_model_is_empty(self):
		sfr_store_init = []
		release_proportion = []
		self.assert_get_flows({}, sfr_store_init, release_proportion, [], [], [])

	def test_get_flows_for_node_count_1_str_count_0(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0)}
		sfr_store_init = []
		release_proportion = []
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [], [], [])

	def test_get_flows_for_node_count_2_str_count_0(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 0),
			2 : make_routing_parameters(downstr = 0),
		}
		sfr_store_init = []
		release_proportion = []
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [], [], [])

	def test_get_flows_for_a_node_count_1_str_count_1(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0, str_flag = 1)}
		sfr_store_init = [0]
		release_proportion = [1]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0], [1], [0])

	def test_get_flows_for_a_node_count_2_str_count_1_with_no_downstream_connections(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 0),
			2 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [0]
		release_proportion = [1]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0], [2], [0])

	def test_get_flows_for_a_node_count_2_str_count_1_with_downstream_connections(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2),
			2 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [0]
		release_proportion = [1]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0], [3], [0])

	def test_get_flows_for_a_node_count_6_str_count_1_with_downstream_connections(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 4),
			2 : make_routing_parameters(downstr = 3),
			3 : make_routing_parameters(downstr = 4),
			4 : make_routing_parameters(downstr = 6),
			5 : make_routing_parameters(downstr = 6),
			6 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [0]
		release_proportion = [1]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0], [63], [0])

	def assert_get_flows(self, sorted_by_ca, sfr_store_init, release_proportion, expected_A, expected_B, expected_sfr_store_total):
		actual_A, actual_B, actual_sfr_total = get_flows_adaptor(sorted_by_ca, sfr_store_init, release_proportion)
		np.testing.assert_array_almost_equal(expected_A, actual_A)
		np.testing.assert_array_almost_equal(expected_B, actual_B)
		np.testing.assert_array_almost_equal(expected_sfr_store_total, actual_sfr_total)

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

def get_flows_adaptor(sorted_by_ca, sfr_store_init, release_proportion):
	swac_seg_dic = {}
	stream_index = 1
	for node_number, params in sorted_by_ca.items():
		if (params[1] == 1):
			swac_seg_dic[node_number] = stream_index
			stream_index += 1

	nodes = len(sorted_by_ca)
	long_list_for_source = [pow(2, x) for x in range(nodes)]
	source = [-1000, -2000, -3000] + long_list_for_source[:nodes]
	index_offset = 3
	actual_A, actual_B, sfr_store = get_attenuated_sfr_flows(sorted_by_ca, swac_seg_dic, nodes, source, index_offset, sfr_store_init, release_proportion)
	return actual_A, actual_B, sfr_store

def get_attenuated_sfr_flows(sorted_by_ca, swac_seg_dic, nodes, source, index_offset, sfr_store_init, release_proportion):
	coalesced_runoff = np.zeros(nodes)
	for node_number, line in sorted_by_ca.items():
		node_index = node_number - 1
		downstr_node_number, str_flag = line[:2]
		has_downstream = downstr_node_number > 0
		downstream_node_index = downstr_node_number - 1
		is_str = str_flag >= 1
		if is_str:
			coalesced_runoff[node_index] += source[node_index + index_offset]
		elif has_downstream:
			coalesced_runoff[downstream_node_index] += source[node_index + index_offset] + coalesced_runoff[node_index]

	stream_cell_count = len(swac_seg_dic)
	coalesced_stream_runoff = np.zeros(stream_cell_count)
	for node_number, stream_cell_number in swac_seg_dic.items():
		node_index = node_number - 1
		stream_cell_index = stream_cell_number - 1
		coalesced_stream_runoff[stream_cell_index] = coalesced_runoff[node_index]

	runoff_result = np.zeros(stream_cell_count)
	flows_result = coalesced_stream_runoff
	actual_sfr_store_total = np.zeros(stream_cell_count)
	return runoff_result, flows_result, actual_sfr_store_total
