import unittest

import numpy as np

class Test_Get_Flows(unittest.TestCase):
	def test_get_flows_for_an_empty_model_is_empty(self):
		self.assert_get_flows({}, [], [])

	def test_get_flows_for_node_count_1_str_count_0(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0)}
		self.assert_get_flows(sorted_by_ca, [], [])

	def test_get_flows_for_a_node_count_1_str_count_1(self):
		sorted_by_ca = {1 : make_routing_parameters(downstr = 0, str_flag = 1)}
		self.assert_get_flows(sorted_by_ca, [0], [0])

	def test_get_flows_for_node_count_2_str_count_0(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 0),
			2 : make_routing_parameters(downstr = 0),
		}
		self.assert_get_flows(sorted_by_ca, [], [])

	def test_get_flows_for_node_count_2_str_count_1_connections_off(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 0),
			2 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [0], [0])

	def test_get_flows_for_node_count_2_str_count_1_connections_on(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2),
			2 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [0], [0])
		# TODO I think that one of the results should have the value 2.

	def test_get_flows_for_node_count_2_str_count_0_connections_on(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2),
			2 : make_routing_parameters(downstr = 0),
		}
		self.assert_get_flows(sorted_by_ca, [], [])

	def test_get_flows_for_node_count_2_str_count_2_connections_off(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [2, 0], [0, 0])
		# TODO I think the result should be [2, 3]

	def test_get_flows_for_node_count_3_str_count_3_connections_off(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 3, str_flag = 1),
			3 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [2, 3, 0], [0, 0, 0])
		# TODO I think the result should be [2, 3, 5]

	def test_get_flows_for_node_count_2_str_count_2_connections_on(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 1, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [2, 0], [0, 0])
		# TODO I think the result should be [2, 5]

	def test_get_flows_for_a_model_size_3_and_3_stream_cells_and_downstream_connections_accumulates_flow(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
			2 : make_routing_parameters(downstr = 3, str_flag = 1),
			3 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [2, 3, 0], [0, 0, 0])
		# TODO I think the result should be [2, 5, 10]

	def test_get_flows_for_a_model_size_3_and_3_stream_cells_and_downstream_connections_in_reverse_order_accumulates_flow(self):
		sorted_by_ca = {
			3 : make_routing_parameters(downstr = 0, str_flag = 1),
			2 : make_routing_parameters(downstr = 3, str_flag = 1),
			1 : make_routing_parameters(downstr = 2, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [0, 3, 2], [0, 0, 0])
		# TODO I think the result should be [5, 3, 2] and that there should be no accumulation because the nodes are in the wrong order.

	def assert_get_flows(self, sorted_by_ca, expected_A, expected_B, explain = False):
		actual_A, actual_B = get_flows_adaptor(sorted_by_ca, explain)
		np.testing.assert_array_almost_equal(expected_A, actual_A)
		np.testing.assert_array_almost_equal(expected_B, actual_B)

	def test_get_flows_for_a_model_size_6_and_3_stream_cells_with_one_cell_contributing_to_each_stream(self):
		sorted_by_ca = {
			1 : make_routing_parameters(downstr = 4),
			2 : make_routing_parameters(downstr = 5),
			3 : make_routing_parameters(downstr = 6),
			4 : make_routing_parameters(downstr = 5, str_flag = 1),
			5 : make_routing_parameters(downstr = 6, str_flag = 1),
			6 : make_routing_parameters(downstr = 0, str_flag = 1),
		}
		self.assert_get_flows(sorted_by_ca, [7, 11, 0], [2, 3, 0])
		# TODO I think the result should be [7, 11, 13], [2, 3, 5].

def get_flows_adaptor(sorted_by_ca, explain = False):
	swac_seg_dic = {}
	stream_index = 1
	for node_number, params in sorted_by_ca.items():
		if (params[1] == 1):
			swac_seg_dic[node_number] = stream_index
			stream_index += 1

	nodes = len(sorted_by_ca)
	nss = len(list(filter(lambda x : x[1] == 1, sorted_by_ca.values())))
	long_list_for_source = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
	source = [-1000, -2000, -3000] + long_list_for_source[:nodes]
	index_offset = 3
	actual_A, actual_B = get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset, explain)
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

def get_flows(sorted_by_ca, swac_seg_dic, nodes, nss, source, index_offset, explain = False):
    result_A = np.zeros((nss))
    result_B = np.zeros((nss))
    done = np.zeros((nodes), dtype=int)

    if explain:
        print()
        print(f"sorted_by_ca = {sorted_by_ca}")
        print(f"swac_seg_dic = {swac_seg_dic}")
        print(f"nodes = {nodes}")
        print(f"nss = {nss}")
        print(f"source = {source}")
        print(f"index_offset = {index_offset}")
        print()

    for node_number, line in sorted_by_ca.items():
        node_index = node_number - 1
        downstr, str_flag = line[:2]
        acc = 0.0
        iteration_number = 0
        do_explain_for(explain, node_number, node_index, downstr, str_flag, acc)
        while downstr > 1:
            str_flag = sorted_by_ca[node_number][1]
            is_str = str_flag >= 1
            is_done = done[node_index] == 1
            stream_cell_index = None

            if is_str:
                stream_cell_index = swac_seg_dic[node_number] - 1

                if is_done:
                    result_B[stream_cell_index] += acc
                    acc = 0.0
                    do_explain(explain, iteration_number, node_number, node_index, downstr, str_flag, is_str, is_done, acc, stream_cell_index, result_A, result_B)
                    break
                else:
                    result_A[stream_cell_index] = source[node_index + index_offset]
                    result_B[stream_cell_index] = acc
                    done[node_index] = 1
                    acc = 0.0

            else:
                if not is_done:
                    acc += max(0.0, source[node_index + index_offset])
                    done[node_index] = 1

            do_explain(explain, iteration_number, node_number, node_index, downstr, str_flag, is_str, is_done, acc, stream_cell_index, result_A, result_B)
            node_number = downstr
            node_index = node_number - 1
            downstr = sorted_by_ca[node_number][0]
            iteration_number += 1
    if explain:
        print(f"acc = {acc}; result_A = {result_A}; result_B = {result_B}")
    return result_A, result_B

def do_explain_for(explain, node_number, node_index, downstr, str_flag, acc):
	if explain:
		print(f"  node_number = {node_number}; node_index = {node_index}; downstr = {downstr}; str_flag = {str_flag}; acc = {acc}")

def do_explain(explain, iteration_number, node_number, node_index, downstr, str_flag, is_str, is_done, acc, stream_cell_index, result_A, result_B):
	if explain:
		if (stream_cell_index is not None):
			suffix = f"; result_A = {result_A[stream_cell_index]}; result_B = {result_B[stream_cell_index]}"
		else:
			suffix = ""
		print(f"  iteration_number = {iteration_number}; node_number = {node_number}; node_index = {node_index}; downstr = {downstr}; str_flag = {str_flag}; is_str = {is_str}; is_done = {is_done}; acc = {acc}" + suffix)
