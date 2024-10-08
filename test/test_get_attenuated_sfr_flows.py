import unittest
import numpy as np
import swacmod.model as m

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

	def test_get_flows_coalesces_into_tow_unrelated_stream_cells(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 3),
			2 : make_routing_parameters(downstr = 4),
			3 : make_routing_parameters(downstr = 5),
			4 : make_routing_parameters(downstr = 6),
			5 : make_routing_parameters(downstr = 0, str_flag = 1),
			6 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [0, 0]
		release_proportion = [1, 1]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0, 0], [21, 42], [0, 0])

	def test_get_flows_for_a_node_count_1_str_count_1_release_proportion_splits_between_flow_and_store(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0, str_flag = 1)}
		sfr_store_init = [0]
		release_proportion = [0.8]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0], [0.8], [0.2])

	def test_get_flows_for_a_node_count_3_str_count_3_release_proportion_splits_between_flow_and_store(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 3, str_flag = 1),
			3 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [0, 0, 0]
		release_proportion = [0.8, 0.5, 0.25]
		# Accumulated flow should be [0.8, 1.4, 1.35], so un-accumulated flow should be [0.8, 0.6, -0.05]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0, 0, 0], [0.8, 0.6, -0.05], [0.2, 1.4, 4.05])

	def test_get_flows_with_different_sfr_store_init(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 3, str_flag = 1),
			3 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [10, 20, 30]
		release_proportion = [0.8, 0.5, 0.25]
		# Accumulated flow should be [5.5, 13.75, 11.9375], so un-accumulated flow should be [8.8, 6.6, -3.05]
		self.assert_get_flows(sorted_by_ca, sfr_store_init, release_proportion, [0, 0, 0], [8.8, 6.6, -3.05], [2.2, 15.4, 37.05])

	def test_get_flows_for_branching_stream_and_dry_cells(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 3),
			2 : make_routing_parameters(downstr = 3),
			3 : make_routing_parameters(downstr = 4),
			4 : make_routing_parameters(downstr = 12, str_flag = 1),
			5 : make_routing_parameters(downstr = 6),
			6 : make_routing_parameters(downstr = 7),
			7 : make_routing_parameters(downstr = 8),
			8 : make_routing_parameters(downstr = 12, str_flag = 1),
			9 : make_routing_parameters(downstr = 12),
			10 : make_routing_parameters(downstr = 12),
			11 : make_routing_parameters(downstr = 12),
			12 : make_routing_parameters(downstr = 16, str_flag = 1),
			13 : make_routing_parameters(downstr = 16),
			14 : make_routing_parameters(downstr = 15),
			15 : make_routing_parameters(downstr = 16),
			16 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		sfr_store_init = [0, 0, 0, 0]
		release_proportion = [0.2, 0.4, 0.6, 0.8]
		# Accumulated flow should be [3, 96, 2363.4, 51042.72], so un-accumulated flow should be [3, 93, 2264.4, 48679.32]
		self.assert_get_flows(
			sorted_by_ca,
			sfr_store_init,
			release_proportion,
			[0, 0, 0, 0],
			[3, 96, 2264.4, 48679.32],
			[12, 144, 1575.6, 12760.68])

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
	actual_A, actual_B, sfr_store = m.get_attenuated_sfr_flows(sorted_by_ca, swac_seg_dic, nodes, source, index_offset, sfr_store_init, release_proportion)
	return actual_A, actual_B, sfr_store
