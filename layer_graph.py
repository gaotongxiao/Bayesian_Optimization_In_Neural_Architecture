import networkx as nx
from enum import Enum
import random

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
    
    def add_node(self, type, num_of_filters=1, stride=2):
        '''
        Return:
            node number
        '''
        self._graph.add_node(self.layer_count, type=type, num_of_filters=num_of_filters, layer_mass=0, stride=stride)
        self.layer_count += 1
        return self.layer_count - 1
    
    def add_edge(self, f, t):
        self._graph.add_edge(f, t)

    def append(self, type, num_of_filters=1, stride=2, append_to=None):
        if append_to is None:
            append_to = self.layer_count - 1
        self.add_edge(append_to, self.add_node(type, num_of_filters, stride))
    
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


    def processing_nodes(self):
        '''
        Return list of tuples, (idx, node)
        '''

        def is_processing_node(node):
            return node['type'] in ['conv3', 'conv5', 'conv7', 'maxpool', 'avgpool', 'fc']

        ret = []
        for idx, node in enumerate(self.get_graph().nodes):
            if is_processing_node(node):
                ret.append(idx, node)

        return ret

    def mut_alt_single(self, portion):
        random_idx, random_node = random.choice(self.is_processing_node())
        num_of_filters = random_node['num_of_filters']
        self.get_graph().nodes[random_idx]['num_of_filters'] = int(num_of_filters*(1+portion))

    def mut_dec_single(self):
        self.mut_alt_single(-1/8)

    def mut_inc_single(self):
        self.mut_alt_single(1/8)

    def mut_alt_en_masse(self, portion):
        num_of_nodes = len(self.processing_nodes())
        rate = 1 + portion
        if  num_of_nodes > 8:
            for random_idx, random_node in random.sample(self.processing_nodes(), int(num_of_nodes/8)):
                num_of_filters = random_node['num_of_filters']
                self.get_graph().nodes[random_idx]['num_of_filters'] = int(num_of_filters*rate)
        else if num_of_nodes > 4:
            for random_idx, random_node in random.sample(self.processing_nodes(), int(num_of_nodes/4)):
                num_of_filters = random_node['num_of_filters']
                self.get_graph().nodes[random_idx]['num_of_filters'] = int(num_of_filters*rate)
        else:
            for random_idx, random_node in random.sample(self.processing_nodes(), int(num_of_nodes/2)):
                num_of_filters = random_node['num_of_filters']
                self.get_graph().nodes[random_idx]['num_of_filters'] = int(num_of_filters*rate)

    def mut_dec_en_masse(self):
        self.mut_alt_en_masse(-1/8)

    def mut_inc_en_masse(self):
        self.mut_alt_en_masse(1/8)


        
            