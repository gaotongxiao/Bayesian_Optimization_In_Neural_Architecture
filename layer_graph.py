import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
import random
import numpy as np

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
        self.conv_layers = [layers.conv3, layers.conv5, layers.conv7]
        self.pool_layers = [layers.maxpool, layers.avgpool]
        self.process_layers = [*self.conv_layers, *self.pool_layers, layers.fc]
        self.decision_layers = [layers.softmax]
        self.iop_layers = [layers.ip, layers.op]
    
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
        new_node = self.add_node(type, num_of_filters, stride)
        self.add_edge(append_to, new_node)
        return new_node
    
    def get_node_attr(self, n, attr='type'):
        return self._graph.node[n][attr]
    
    def get_graph(self):
        return self._graph

    def get_nodes(self):
        return nx.topological_sort(self._graph)

    def get_node(self, idx):
        return self.get_graph().nodes[idx]

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
            if self.get_node_attr(node) in self.iop_layers:
                ipop.append(node)
            elif self.get_node_attr(node) in self.decision_layers:
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
        Return list of node idx
        '''

        def is_processing_node(node_idx):
            return self.get_node(node_idx)['type'] in self.process_layers

        return [node_idx for node_idx in self.get_graph().nodes if is_processing_node(node_idx)]

    def mut_alt_single(self, portion):
        random_node = self.get_node(random.choice(self.processing_nodes()))
        num_of_filters = random_node['num_of_filters']
        random_node['num_of_filters'] = int(num_of_filters*(1+portion))

    def mut_dec_single(self):
        self.mut_alt_single(-1/8)

    def mut_inc_single(self):
        self.mut_alt_single(1/8)

    def mut_alt_en_masse(self, portion):
        num_of_nodes = len(self.processing_nodes())
        rate = 1 + portion
        if num_of_nodes > 8:
            num_of_mut = int(num_of_nodes/8)
        elif num_of_nodes > 4:
            num_of_mut = int(num_of_nodes/4)
        else :
            num_of_mut = int(num_of_nodes/2)
        for random_node_idx in random.sample(self.processing_nodes(), int(num_of_mut)):
            random_node = self.get_node(random_node_idx)
            num_of_filters = random_node['num_of_filters']
            random_node['num_of_filters'] = int(num_of_filters*rate)

    def mut_dec_en_masse(self):
        self.mut_alt_en_masse(-1/8)

    def mut_inc_en_masse(self):
        self.mut_alt_en_masse(1/8)

    def show_graph(self, node_size=1000):
        plt.figure()
        labels = {}
        for n, t in self._graph.nodes(data=True):
            labels[n] = str(t['type']) + ' ' + str(t['stride']) + ' ' + str(t['num_of_filters'])
        nx.draw_kamada_kawai(self._graph, labels=labels, node_size=node_size)
        # plt.show()

    def mut_skip(self):
        def random_pick():
            A = random.randint(1, self.layer_count-1)
            while self.get_node_attr(A) not in self.process_layers:
                A = random.randint(1, self.layer_count-1)
            B = random.randint(1, self.layer_count-1)
            while self.get_node_attr(B) not in self.process_layers:
                B = random.randint(1, self.layer_count-1)
            for n in self.get_nodes():
                if A == n:
                    break
                if B == n:
                    B = A
                    A = n
                    break
            return [A, B]
        i = 0
        nodes = random_pick()
        while self._graph.has_edge(*nodes) and i < 20:
            i += 1
            nodes = random_pick()
        path = nx.shortest_path(self._graph, source=nodes[0], target=nodes[1])
        pool_counter = 0
        for node in path[1:-1]:
            if self.get_node_attr(node) in self.pool_layers:
                pool_counter += 1
        while pool_counter != 0:
            nodes[0] = self.append(layers.avgpool, append_to=nodes[0])
            pool_counter -= 1
        self.add_edge(nodes[0], nodes[1])
        

    def mut_swap_label(self):
        '''
        Can I pick softmax or change to softmax?
        '''
        node = random.randint(1, self.layer_count-1)
        while self.get_node_attr(node) not in self.process_layers:
            node = random.randint(1, self.layer_count-1)
        layers_list = list(layers)
        type = random.choice(layers_list)
        while type not in self.process_layers:
            type = random.choice(layers_list)
        if type in [*self.conv_layers, layers.fc]:
            num_of_filters = np.random.choice([64, 128, 256, 512], 1, p=[0.4, 0.3, 0.2, 0.1])[0]
            stride = random.choice([1, 2]) if type != layers.fc else 2
        else:
            num_of_filters = 1
            stride = 2
        self._graph.node[node]['stride'] = stride
        self._graph.node[node]['num_of_filters'] = num_of_filters
        self._graph.node[node]['type'] = type

    
    def mut_wedge_layer(self):
        edges = self._graph.edges()
        edge = random.choice(list(edges))
        while self.get_node_attr(edge[0]) not in self.process_layers and self.get_node_attr(edge[1]) not in self.process_layers:
            edge = random.choice(list(edges))
        print("hi")
        layers_list = list(layers)
        type = random.choice(layers_list)
        while type not in self.process_layers:
            type = random.choice(layers_list)
        if type in [*self.conv_layers, layers.fc]:
            num_of_filters = int((self.get_node_attr(
                edge[0], 'num_of_filters') + self.get_node_attr(edge[1], 'num_of_filters')) / 2)
            if num_of_filters % 2: num_of_filters += 1
            if num_of_filters < 16:
                num_of_filters = np.random.choice([64, 128, 256, 512], 1, p=[0.4, 0.3, 0.2, 0.1])[0]
            stride = random.choice([1, 2]) if type != layers.fc else 2
        else:
            num_of_filters = 1
            stride = 2
        self._graph.remove_edge(*edge)
        new_node = self.append(type, num_of_filters, stride, edge[0])
        self.add_edge(new_node, edge[1])
        
