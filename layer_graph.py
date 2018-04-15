import networkx as nx
from enum import Enum

layers_type_num = 9
layers = Enum('layers', ('conv3', 'conv5', 'conv7', 'maxpool', 'avgpool', 'fc', 'ip', 'op', 'softmax'))

class layer_graph(object):
    '''
    Don't need to speicify input layer
    '''
    def __init__(self, input_unit):
        self._graph = nx.DiGraph()
        self.layer_count = 0
        self.input_unit = input_unit
        self.total_lm = 0
    
    def add_node(self, type, num_of_filters=1):
        '''
        Return:
            node number
        '''
        self._graph.add_node(self.layer_count, type=type, num_of_filters=num_of_filters, layer_mass=0)
        self.layer_count += 1
        return self.layer_count - 1
    
    def add_edge(self, f, t):
        self._graph.add_edge(f, t)

    def append(self, type, num_of_filters=1, append_to=None):
        if append_to is None:
            append_to = self.layer_count - 1
        self.add_edge(append_to, self.add_node(type, num_of_filters))
    
    def get_node_attr(self, n, attr='type'):
        return self._graph.node[n][attr]
    
    def get_graph(self):
        return self._graph

    def get_nodes(self):
        return nx.topological_sort(self._graph)

    def update_lm(self, zeta1=0.1, zeta2=0.1):
        '''
        Args:
            zeta1: for ip, op
            zeta2: for decision layers
        '''
        nodes = self.get_nodes()
        ipop = []
        dl = []
        pl_lm = 0
        for node in nodes:
            if self._graph.node[node]['type'] == layers.ip or self._graph.node[node]['type'] == layers.op:
                ipop.append(node)
            elif self._graph.node[node]['type'] == layers.softmax:
                dl.append(node)
            else:
                total_filters = 0
                for n in self._graph.predecessors(node):
                    total_filters += self._graph.node[n]['num_of_filters']
                k = 0.1 if self._graph.node[node]['type'] == layers.fc else 1
                self._graph.node[node]['layer_mass'] = k * total_filters * self._graph.node[node]['num_of_filters']
                pl_lm += self._graph.node[node]['layer_mass']
        self.total_lm  = pl_lm
        for node in ipop:
            self._graph.node[node]['layer_mass'] = zeta1 * pl_lm
            self.total_lm += self._graph.node[node]['layer_mass']
        for node in dl:
            self._graph.node[node]['layer_mass'] = zeta2 * pl_lm / len(dl)
            self.total_lm += self._graph.node[node]['layer_mass']
    
    def get_num_layers(self):
        return self.layer_count
    
    def get_total_mass(self):
        return self.total_lm
            
        
            