import unittest
import networkx as nx
import swacmod.feature_flags as ff
import swacmod.networkx_adaptor as networkx_adaptor

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

	def test_build_graph_adds_ca_to_nodes_when_use_natproc_is_true(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(3)
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc = True)
		self.assertEqual(10, graph.nodes[1]["ca"])
		self.assertEqual(20, graph.nodes[2]["ca"])
		self.assertEqual(30, graph.nodes[3]["ca"])

	def test_build_graph_does_not_add_ca_to_nodes_when_use_natproc_is_false(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(3)
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc = False)
		self.assertTrue("ca" not in graph.nodes[1])
		self.assertTrue("ca" not in graph.nodes[2])
		self.assertTrue("ca" not in graph.nodes[3])

	def test_build_graph_does_not_add_masked_nodes_when_use_natproc_is_false(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(3)
		mask = [0, 1, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc = False)
		self.assertEqual([2], list(graph.nodes))

	def test_build_graph_does_not_add_masked_nodes_when_use_natproc_is_true(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(3)
		mask = [0, 1, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc = True)
		self.assertEqual([2], list(graph.nodes))

	def test_build_graph_adds_edges(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2, str_flag = 1),
			2: make_sorted_by_ca_line(2, str_flag = 1),
		}
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual(1, graph.number_of_edges())
		self.assertTrue(graph.has_edge(1, 2))

	def test_build_graph_adds_multiple_edges(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(4)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2, str_flag = 1),
			2: make_sorted_by_ca_line(2, downstr = 4, str_flag = 1),
			3: make_sorted_by_ca_line(3, downstr = 4, str_flag = 1),
			4: make_sorted_by_ca_line(4, str_flag = 1),
		}
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual(3, graph.number_of_edges())
		self.assertTrue(graph.has_edge(1, 2))
		self.assertTrue(graph.has_edge(2, 4))
		self.assertTrue(graph.has_edge(3, 4))

	def test_build_graph_adds_masked_downstream_nodes_when_required_for_edges(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2, str_flag = 1),
			2: make_sorted_by_ca_line(2),
		}
		mask = [1, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual([1, 2], list(graph.nodes))

	def test_build_graph_does_not_add_masked_downstream_nodes_when_required_for_edges(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2),
			2: make_sorted_by_ca_line(2, str_flag = 1),
		}
		mask = [0, 1]
		graph = build_graph(nnodes, sorted_by_ca, mask)
		self.assertEqual([2], list(graph.nodes))

	def test_build_graph_adds_added_masked_nodes_include_ca_when_use_natproc_is_true(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2, str_flag = 1),
			2: make_sorted_by_ca_line(2),
		}
		mask = [1, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc=True)
		self.assertEqual(20, graph.nodes[2]["ca"])

	def test_build_graph_adds_added_masked_nodes_does_not_include_ca_when_use_natproc_is_false(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2, str_flag = 1),
			2: make_sorted_by_ca_line(2),
		}
		mask = [1, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc=False)
		self.assertTrue("ca" not in graph.nodes[2])

	def test_build_graph_converts_downstream_to_int_when_natproc_is_true(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = "2", str_flag = 1),
			2: make_sorted_by_ca_line(2),
		}
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc=True)
		self.assertTrue(graph.has_edge(1, 2))

	def test_build_graph_does_not_add_extra_downstream_nodes_when_natproc_is_false(self):
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2),
			2: make_sorted_by_ca_line(2),
		}
		mask = [0, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc=False)
		self.assertEqual([], list(graph.nodes))

	def test_build_graph_adds_extra_downstream_nodes_when_natproc_is_true(self):
		"""
		I am almost certain that this represents a bug.
		This means that all the dry cells will be included in the graph even though they have no edges.
		"""
		nnodes, sorted_by_ca, mask = make_args_for_node_count(2)
		sorted_by_ca = {
			1: make_sorted_by_ca_line(1, downstr = 2),
			2: make_sorted_by_ca_line(2),
		}
		mask = [0, 0]
		graph = build_graph(nnodes, sorted_by_ca, mask, use_natproc=True)
		self.assertEqual([2], list(graph.nodes))

def make_args_for_node_count(node_count):
	nnodes = node_count
	sorted_by_ca = {}
	for node_index in range(node_count):
		node_number = node_index + 1
		line = make_sorted_by_ca_line(node_number, str_flag = 1)
		sorted_by_ca[node_number] = line
	mask = [1] * node_count
	return nnodes, sorted_by_ca, mask

def make_sorted_by_ca_line(node_number, downstr = -1, str_flag = 0):
	node_mf = None
	length = None
	ca = 10 * node_number
	z = None
	bed_thk = None
	str_k = None
	depth = None
	width = None
	return (downstr, str_flag, node_mf, length, ca, z, bed_thk, str_k, depth, width)

def build_graph(nnodes, sorted_by_ca, mask, di=True, use_natproc = None):
    if use_natproc is None:
        use_natproc = ff.use_natproc

    nodes = {}
    for node_index in range(nnodes):
        if mask[node_index] == 1:
            node_number = node_index + 1
            if use_natproc:
                nodes[node_number] = {"ca":sorted_by_ca[node_number][4]}
            else:
                nodes[node_number] = {}

    edges = []
    for node_swac, line in sorted_by_ca.items():
        downstr = int(line[0])
        if downstr > 0:
            if use_natproc:
                if downstr not in nodes:
                    nodes[downstr] = {"ca":sorted_by_ca[downstr][4]}
            if mask[node_swac-1] == 1:
                edges.append((node_swac, downstr))

    if di:
        return networkx_adaptor.make_directed_graph(nodes, edges)
    else:
        return networkx_adaptor.make_undirected_graph(nodes, edges)
