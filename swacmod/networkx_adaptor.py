import networkx as nx

def make_directed_graph(nodes, edges):
	return _populate_graph(nx.DiGraph(), nodes, edges)

def make_undirected_graph(nodes, edges):
	return _populate_graph(nx.Graph(), nodes, edges)

def _populate_graph(G, nodes, edges):
	for node, attr in nodes.items():
		G.add_node(node, **attr)
	for e in edges:
		G.add_edge(e[0], e[1])
	return Networkx_Adaptor_Graph(G)

def shortest_path_length(G, source=None, target=None):
	return nx.shortest_path_length(G.G, source, target)

def neighbors(G, node):
	return G.G.neighbors(node)

def in_degree(G, node):
	return G.G.in_degree(node)

class Networkx_Adaptor_Graph:
	def __init__(self, G):
		self.G = G

	def nodes(self):
		return list(self.G.nodes())

	def out_degree(self, node):
		return self.G.out_degree(node)

	def in_degree(self, node):
		return self.G.in_degree(node)
