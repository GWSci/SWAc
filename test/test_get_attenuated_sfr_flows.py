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
	nodes_ca_order = []
	stream_ca_order = []
	for node_number, line in sorted_by_ca.items():
		node_index = node_number - 1
		downstr_node_number, str_flag = line[:2]
		downstream_node_index = downstr_node_number - 1
		nodes_ca_order.append((node_index, downstream_node_index, str_flag))
		if str_flag >= 1:
			stream_number = swac_seg_dic[node_number]
			stream_index = stream_number - 1
			if downstream_node_index >= 0:
				downstream_stream_number = swac_seg_dic[downstr_node_number]
				downstream_stream_index = downstream_stream_number - 1
			else:
				downstream_stream_index = -1
			stream_ca_order.append((node_index, stream_index, downstream_stream_index))

	coalesced_runoff = np.zeros(nodes)
	for node_index, downstream_node_index, str_flag in nodes_ca_order:
		if str_flag >= 1:
			coalesced_runoff[node_index] += source[node_index + index_offset]
		elif downstream_node_index >= 0:
			coalesced_runoff[downstream_node_index] += source[node_index + index_offset] + coalesced_runoff[node_index]

	stream_count = len(swac_seg_dic)
	coalesced_stream_runoff = np.zeros(stream_count)
	for node_index, stream_index, _ in stream_ca_order:
		coalesced_stream_runoff[stream_index] = coalesced_runoff[node_index]

	sfr_store_total = sfr_store_init + coalesced_stream_runoff

	sfr_released = np.zeros(stream_count)
	for _, index, downstream_index in stream_ca_order:
		sfr_released[index] = sfr_store_total[index] * release_proportion[index]
		if downstream_index >= 0:
			sfr_store_total[downstream_index] += sfr_released[index]

	sfr_store_total = sfr_store_total - sfr_released

	de_accumulated_flows = np.copy(sfr_released)
	for _, index, downstream_index in stream_ca_order:
		if downstream_index >= 0:
			de_accumulated_flows[downstream_index] -= sfr_released[index]

	runoff_result = np.zeros(stream_count)
	return runoff_result, de_accumulated_flows, sfr_store_total
