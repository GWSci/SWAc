import networkx as nx

def make_directed_graph(nodes, edges):
	G = nx.DiGraph()
	return _populate_graph(G, nodes, edges)

def make_undirected_graph(nodes, edges):
	G = nx.Graph()
	return _populate_graph(G, nodes, edges)

def _populate_graph(G, nodes, edges):
	for node, attr in nodes.items():
		G.add_node(node, **attr)
	for e in edges:
		G.add_edge(e[0], e[1])
	return G