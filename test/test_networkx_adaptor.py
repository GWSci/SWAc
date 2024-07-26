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
		actual = nx.shortest_path_length(G, target = 3)
		expected = {
			1: 2,
			2: 1,
			3: 0,
		}
		self.assertEqual(expected, actual)

	def test_neighbours(self):
		G = make_graph_for_shortest_path_length()
		self.assertEqual([2], list(nx.neighbors(G, 1)))
		self.assertEqual([3, 4], list(nx.neighbors(G, 2)))
		self.assertEqual([], list(nx.neighbors(G, 4)))

	def test_in_degree(self):
		G = make_graph_for_shortest_path_length()
		self.assertEqual(0, nx.in_degree(G, 1))
		self.assertEqual(1, nx.in_degree(G, 2))
		self.assertEqual(1, nx.in_degree(G, 3))
		self.assertEqual(2, nx.in_degree(G, 4))

def make_graph_for_shortest_path_length():
		edges = [
			(1, 2),
			(2, 3),
			(3, 4),
			(2, 4),
		]
		return nx.make_directed_graph({}, edges)
