import networkx as nx
import numpy as np

def getDistance(graph):
  '''
  Compute path length

  Args: 
      graph: layer graph (Assume that it has only one input layer and one output layer)

  Return:
      rw_ip: random walk distance for input layer
      rw_op: random walk distance for output layer
      sp_ip: shortest path for input layer
      sp_op: shortest path for output layer
      lp_ip: longest path for input layer
      lp_op: longest path for output layer
  '''
  rw_ip = {}
  rw_op = {}
  sp_ip = {}
  sp_op = {}
  lp_ip = {}
  lp_op = {}

  nodes = list(nx.topological_sort(graph))

  rw_ip[nodes[0]] = 0
  sp_ip[nodes[0]] = 0
  lp_ip[nodes[0]] = 0
  rw_op[nodes[-1]] = 0
  sp_op[nodes[-1]] = 0
  lp_op[nodes[-1]] = 0

  for node in nodes[1:]:
    data = []
    for p in graph.predecessors(node):
      data.append(rw_ip[p])
    rw_ip[node] = 1 + int(np.mean(data))
    sp_ip[node] = 1 + np.min(data)
    lp_ip[node] = 1 + np.max(data)
  
  for node in reversed(nodes[:-1]):
    data = []
    for p in graph.successors(node):
      data.append(rw_op[p])
    rw_op[node] = 1 + int(np.mean(data))
    sp_op[node] = 1 + np.min(data)
    lp_op[node] = 1 + np.max(data)

  return rw_ip, sp_ip, lp_ip, rw_op, sp_op, lp_op

'''
G = nx.DiGraph()
G.add_node(1, in_dim=15)
G.add_node(2, in_dim=15)
G.add_node(3, in_dim=15)
G.add_node(4, in_dim=15)
G.add_node(5, in_dim=15)
G.add_edge(1, 2)
G.add_edge(3, 1)
G.add_edge(3, 4)
G.add_edge(4, 5)
G.add_edge(5, 2)
print(list(G.successors(1)))
print(G.out_degree(1))
print(list(G.predecessors(2)))
print(G.nodes[1])
print(list(G.successors(1)))
print(list(nx.topological_sort(G)))
print(getDistance(G))
'''