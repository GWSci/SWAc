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
	return G

def shortest_path_length(G, source=None, target=None):
	return nx.shortest_path_length(G, source, target)

def neighbors(G, node):
	return G.neighbors(node)

def in_degree(G, node):
	return G.in_degree(node)

def out_degree(G, node):
	return G.out_degree(node)
