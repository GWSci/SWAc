import unittest
import networkx as nx
import swacmod.feature_flags as ff

class Test_Build_Graph(unittest.TestCase):
	def test_build_empty_graph(self):
		nnodes = None
		sorted_by_ca = None
		mask = None
		di = None
		graph = build_graph(nnodes, sorted_by_ca, mask, di)
		self.assertEqual(0, graph.number_of_nodes())
		self.assertEqual(0, graph.number_of_edges())

	def test_build_graph_creates_a_directed_graph_when_di_is_True(self):
		nnodes = None
		sorted_by_ca = None
		mask = None
		di = True
		graph = build_graph(nnodes, sorted_by_ca, mask, di)
		self.assertEqual("<class 'networkx.classes.digraph.DiGraph'>", str(type(graph)))

def build_graph(nnodes, sorted_by_ca, mask, di=True):
    # if di:
    G = nx.DiGraph()
    # else:
    #     G = nx.Graph()
    # for node in range(1, nnodes + 1):
    #     if ff.use_natproc:
    #         if mask[node-1] == 1: #  and sorted_by_ca[node][4] > 0.0:
    #             G.add_node(node, ca=sorted_by_ca[node][4])
    #     else:
    #         if mask[node-1] == 1:
    #             G.add_node(node)
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
