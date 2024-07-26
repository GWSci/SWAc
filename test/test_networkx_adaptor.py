import unittest
import swacmod.networkx_adaptor as nx

class Test_Networkx_Adaptor(unittest.TestCase):
	def test_shortest_path_length_with_source_and_target_specified(self):
		G = make_graph_for_shortest_path_length()
		actual = nx.shortest_path_length(G, source = 1, target = 4)
		self.assertEqual(2, actual)

	def test_shortest_path_length_with_only_source_specified(self):
		G = make_graph_for_shortest_path_length()
		actual = nx.shortest_path_length(G, source = 1)
		expected = {
			1: 0,
			2: 1,
			3: 2,
			4: 2,
		}
		self.assertEqual(expected, actual)

	def test_shortest_path_length_with_only_source_specified_and_not_all_nodes_reachable(self):
		G = make_graph_for_shortest_path_length()
		actual = nx.shortest_path_length(G, source = 2)
		expected = {
			2: 0,
			3: 1,
			4: 1,
		}
		self.assertEqual(expected, actual)

	def test_shortest_path_length_with_only_target_specified(self):
		G = make_graph_for_shortest_path_length()
		actual = nx.shortest_path_length(G, target = 4)
		expected = {
			1: 2,
			2: 1,
			3: 1,
			4: 0,
		}
		self.assertEqual(expected, actual)

	def test_shortest_path_length_with_only_target_specified_and_not_all_nodes_reachable(self):
		G = make_graph_for_shortest_path_length()
		actual = G.shortest_path_length(target = 3)
		expected = {
			1: 2,
			2: 1,
			3: 0,
		}
		self.assertEqual(expected, actual)

	def test_neighbours(self):
		G = make_graph_for_shortest_path_length()
		self.assertEqual([2], list(G.neighbors(1)))
		self.assertEqual([3, 4], list(G.neighbors(2)))
		self.assertEqual([], list(G.neighbors(4)))

	def test_in_degree(self):
		G = make_graph_for_shortest_path_length()
		self.assertEqual(0, G.in_degree(1))
		self.assertEqual(1, G.in_degree(2))
		self.assertEqual(1, G.in_degree(3))
		self.assertEqual(2, G.in_degree(4))

	def test_out_degree(self):
		G = make_graph_for_shortest_path_length()
		self.assertEqual(1, G.out_degree(1))
		self.assertEqual(2, G.out_degree(2))
		self.assertEqual(1, G.out_degree(3))
		self.assertEqual(0, G.out_degree(4))

	def test_nodes(self):
		G = make_graph_for_shortest_path_length()
		self.assertEqual([1, 2, 3, 4], G.nodes())

def make_graph_for_shortest_path_length():
		edges = [
			(1, 2),
			(2, 3),
			(3, 4),
			(2, 4),
		]
		return nx.make_directed_graph({}, edges)
