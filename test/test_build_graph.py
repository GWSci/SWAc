import unittest
import networkx as nx
import swacmod.feature_flags as ff

class Test_Build_Graph(unittest.TestCase):
	def test_build_empty_graph(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(0)
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual(0, graph.number_of_nodes())
		self.assertEqual(0, graph.number_of_edges())

	def test_build_graph_creates_a_directed_graph_when_di_is_omitted(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(0)
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual("<class 'networkx.classes.digraph.DiGraph'>", str(type(graph)))

	def test_build_graph_creates_a_directed_graph_when_di_is_True(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(0)
		di = True
		graph = build_graph(nnodes, sorted_by_ca, mask, di)
		self.assertEqual("<class 'networkx.classes.digraph.DiGraph'>", str(type(graph)))

	def test_build_graph_creates_a_non_directed_graph_when_di_is_False(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(0)
		di = False
		graph = build_graph(nnodes, sorted_by_ca, mask, di)
		self.assertEqual("<class 'networkx.classes.graph.Graph'>", str(type(graph)))

	def test_build_graph_adds_one_node(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(1)
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual(1, graph.number_of_nodes())
		self.assertEqual(0, graph.number_of_edges())

	def test_build_graph_adds_multiple_nodes(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(3)
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual(3, graph.number_of_nodes())
		self.assertEqual(0, graph.number_of_edges())

def make_args_for_node_count(node_count):
	nnodes = node_count
	sorted_by_ca = {}
	for node_index in range(node_count):
		node_number = node_index + 1
		line = node_number
		sorted_by_ca[node_number] = line
	mask = [1] * node_count
	return nnodes, sorted_by_ca, mask

def make_sorted_by_ca_line(node_number, downstr = -1, str_flag = 0):
	node_mf = None
	length = None
	ca = None
	z = None
	bed_thk = None
	str_k = None
	depth = None
	width = None
	return (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width)

def build_graph(nnodes, sorted_by_ca, mask, di=True, use_natproc = None):
    if use_natproc is None:
        use_natproc = ff.use_natproc
    if di:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for node in range(1, nnodes + 1):
    #     if ff.use_natproc:
    #         if mask[node-1] == 1: #  and sorted_by_ca[node][4] > 0.0:
    #             G.add_node(node, ca=sorted_by_ca[node][4])
    #     else:
    #         if mask[node-1] == 1:
                G.add_node(node)
    # for node_swac, line in sorted_by_ca.items():
    #     if ff.use_natproc:
    #         downstr = int(line[0])
    #     else:
    #         downstr = line[0]
    #     if downstr > 0:
    #         if ff.use_natproc:
    #             if downstr not in G.nodes:
    #                 G.add_node(downstr, ca=sorted_by_ca[downstr][4])
    #         else:
    #             pass
    #         if mask[node_swac-1] == 1:
    #             G.add_edge(node_swac, downstr)
    return G
