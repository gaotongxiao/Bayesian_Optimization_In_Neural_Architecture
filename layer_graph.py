import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
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

    #remove designated pool layer - helper
    def remove_a_pool(self, node):
        for parent in self._graph.predecessors(node):
            if sum(1 for _ in self._graph.successors(parent)) == 1:
                #only child for this parent
                self.add_edge(parent, first(self._graph.successors(node)))#connect parent u with del's child
        for child in self._graph.successors(node):
            if sum(1 for _ in self._graph.predecessors(child)) == 1:
                #only parent for this child
                self.add_edge(first(self._graph.predecessors(node), child))#connect child u with del's parent
        self._graph.remove_node(node)

    #remove pool recursively by going up - helper
    def remove_pool(self, node):
        #base case
        if node['type'] == layers.maxpool or node['type'] == layers.avgpool:
            remove_a_pool(node)
        else:
            for parent in self._graph.predecessors(node):
                remove_pool(parent)

    def mut_dup_path(self):
        #king
        pass

    def mut_remove_layer(self):
        #king
        is_pool = False
        nodes = self.get_nodes()
        while True:
            pick = random.randint(0, self.layer_count-1)
#            if nodes[pick]['type'] == layers.maxpool or nodes[pick]['type'] == layers.avgpool:
#                is_pool = True
#                break
            if nodes[pick]['type'] == layers.conv3 or nodes[pick]['type'] == layers.conv5 or nodes[pick]['type'] == layers.conv7:
                break
        for parent in self._graph.predecessors(nodes[pick]):
            if sum(1 for _ in self._graph.successors(parent)) == 1:
                #only child for this parent
                self.add_edge(parent, first(self._graph.successors(nodes[pick])))#connect parent u with del's child
        for child in self._graph.successors(nodes[pick]):
            if sum(1 for _ in self._graph.predecessors(child)) == 1:
                #only parent for this child
                self.add_edge(first(self._graph.predecessors(nodes[pick])), child)#connect child u with del's parent
        #remove pool requires update for other paths
#        if is_pool:
            #find the closest gathering point for every child
#            for child in self._graph.successors(nodes[pick])
#            while True:
        self._graph.remove_node(nodes[pick])

    def show_graph(self, node_size=1000):
        plt.subplot(121)
        labels = {}
        for n, t in self._graph.nodes(data=True):
            labels[n] = str(t['type']) + ' ' + str(t['stride']) + ' ' + str(t['num_of_filters'])
        nx.draw_kamada_kawai(self._graph, labels=labels, node_size=node_size)
        plt.show()
