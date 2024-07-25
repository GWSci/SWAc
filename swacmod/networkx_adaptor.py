import networkx as nx

def make_directed_graph(nodes, edges):
	G = nx.DiGraph()
	for node, attr in nodes.items():
		G.add_node(node, **attr)

	for e in edges:
		G.add_edge(e[0], e[1])
	return G

def make_undirected_graph(nodes, edges):
	G = nx.Graph()

	for node, attr in nodes.items():
		G.add_node(node, **attr)

	for e in edges:
		G.add_edge(e[0], e[1])
	return G
