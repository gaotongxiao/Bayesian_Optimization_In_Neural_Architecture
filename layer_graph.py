import networkx as nx
from enum import Enum

layers = Enum('layers', ('conv3', 'conv5', 'maxpool', 'avgpool', 'fc', 'ip', 'op'))

class layer_graph(object):
    def __init__(self, input_unit):
        self._graph = nx.DiGraph()
        self._graph.add_node(0, num_of_filters=1, type=layers.ip, layer_mass=0)
        self.layer_count = 1
        self.input_unit = input_unit
    
    def add_node(self, type, num_of_filters):
        '''
        Return:
            node number
        '''
        self._graph.add_node(self.layer_count, type=type, num_of_filters=num_of_filters, layer_mass=0)
        self.layer_count += 1
        return self.layer_count - 1
    
    def add_edge(self, f, t):
        '''
        Ignore efficiency
        '''
        self._graph.add_edge(f, t)
        total_filters = 0
        for n in self._graph.predecessors(t):
            total_filters += self._graph.node[n]['num_of_filters']
        self._graph.node[t]['layer_mass'] = total_filters * self._graph.node[t]['num_of_filters']
    
    def get_node(self, n):
        return self._graph.node[n]


'''
G = layer_graph(1)
G.add_node(layers.conv3, 16)
G.add_node(layers.conv3, 16)
G.add_node(layers.conv3, 32)
G.add_edge(1, 3)
G.add_edge(2, 3)
print(G.get_node(3))
'''