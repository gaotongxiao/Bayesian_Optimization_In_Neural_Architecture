import networkx as nx
import numpy as np
import layer_graph as lg
from layer_graph import layers

def get_path_length(graph):
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

def get_lmm_matrix(g1, g2):
    
    def symmetrize(mat):
        '''
            Abandon lower triangle
        '''
        M = np.triu(mat)
        return M + M.T
    '''
    Construct cost matrix
    When indexing M by enum type, remember minus it by 1 (enum starts from 1)
    '''
    M = np.ones((lg.layers_type_num, lg.layers_type_num)) * 3 # Inf should be any value larger than 2
    np.fill_diagonal(M, 0)
    M[layers.conv3.value - 1, layers.conv5.value - 1] = 0.2
    M[layers.conv3.value - 1, layers.conv7.value - 1] = 0.3
    M[layers.conv5.value - 1, layers.conv7.value - 1] = 0.2
    M[layers.maxpool.value - 1, layers.avgpool.value - 1] = 0.25
    M = symmetrize(M)
    #Construct penality matrix
    C = np.zeros((g1.get_num_layers(), g2.get_num_layers()))
    for c1, n1 in enumerate(g1.get_nodes()):
        for c2, n2 in enumerate(g2.get_nodes()):
            C[c1, c2] = M[g1.get_node_attr(n1).value - 1, g2.get_node_attr(n2).value - 1]
    return symmetrize(C)

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
print(getPathLength(G))
'''