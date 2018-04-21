import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
LAYERS = Enum('layers', ('conv3', 'conv5', 'conv7', 'maxpool', 'avgpool', 'fc', 'ip', 'op', 'softmax'))
from distance import get_distance

layers_type_num = 9

layer_graph_count = 0
layer_graph_table = []

class Layer_graph(object):
    '''
    Don't need to speicify input layer
    '''
    def __init__(self, input_unit):
        global layer_graph_table, layer_graph_count
        self.id = layer_graph_count
        layer_graph_count += 1
        layer_graph_table.append(self)
        self._graph = nx.DiGraph()
        self.layer_count = 0
        self.input_unit = input_unit
        self.total_lm = 0

        self.conv_layers = [LAYERS.conv3, LAYERS.conv5, LAYERS.conv7]
        self.pool_layers = [LAYERS.maxpool, LAYERS.avgpool]
        self.process_layers = [*self.conv_layers, *self.pool_layers, LAYERS.fc]
        self.decision_layers = [LAYERS.softmax]
        self.iop_layers = [LAYERS.ip, LAYERS.op]


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

    def remove_edge(self, f, t):
        self._graph.remove_edge(f, t)

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

    def finish(self, zeta1=0.1, zeta2=0.1):
        '''
        Args:
            zeta1: for ip, op
            zeta2: for decision layers
        '''
        global layer_graph_table
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
                k = 0.1 if self._graph.node[node]['type'] == LAYERS.fc else 1
                self._graph.node[node]['layer_mass'] = k * total_filters * self._graph.node[node]['num_of_filters']
                pl_lm += self._graph.node[node]['layer_mass']
        self.total_lm  = pl_lm
        for node in ipop:
            self._graph.node[node]['layer_mass'] = zeta1 * pl_lm
            self.total_lm += self._graph.node[node]['layer_mass']
        for node in dl:
            self._graph.node[node]['layer_mass'] = zeta2 * pl_lm / len(dl)
            self.total_lm += self._graph.node[node]['layer_mass']
        # update distance
        for graph in layer_graph_table:
            get_distance(graph, self, update=True)

    #remove designated pool layer - helper
    def remove_a_pool(self, node):
        print('removing...')
        for parent in self._graph.predecessors(node):
            if sum(1 for _ in self._graph.successors(parent)) == 1:
                #only child for this parent
                self.add_edge(parent, next(self._graph.successors(node)))#connect parent u with del's child
        for child in self._graph.successors(node):
            if sum(1 for _ in self._graph.predecessors(child)) == 1:
                #only parent for this child
                self.add_edge(next(self._graph.predecessors(node)), child)#connect child u with del's parent
        self._graph.remove_node(node)
        self.layer_count -= 1

    #remove pool recursively by going up - helper
    def remove_pool(self, node):
        #base case
        if self._graph.node[node]['type'] == LAYERS.maxpool or self._graph.node[node]['type'] == LAYERS.avgpool:
            self.remove_a_pool(node)
        #reach the root
        elif node == 0:
            return
        else:
            tmp_parents = list(self._graph.predecessors(node))
            for parent in tmp_parents:
                self.remove_pool(parent)

    def update_pool(self, after_add):
        '''
        Args:
            after_add: being called after add_layer or not
        '''
        #bfs_edges = list(nx.bfs_edges(self._graph, list(self.get_nodes())[0]))
        #bfs_cnt = np.zeros(len(bfs_edges))#record the num_pool from source to this node
        topo_nodes = list(self.get_nodes())
        topo_cnt = np.zeros(max(topo_nodes)+1)
        is_updated = False
        for update_node in topo_nodes[1:]:#skip the first node
        #for tpl in bfs_edges:
        #    update_node = tpl[1]
            #remove pool
            if not after_add:
                parents = list(self._graph.predecessors(update_node))
                if len(parents)==0:
                    continue
                pool_cnt = np.zeros(len(parents))
                i = 0
                for parent in parents:
                    if self._graph.node[parent]['type'] == LAYERS.maxpool or self._graph.node[parent]['type'] == LAYERS.avgpool:
                        pool_cnt[i] = topo_cnt[parent]+1
                    else:
                        pool_cnt[i] = topo_cnt[parent]
                    i+=1
                #remove pool recursively
                i = 0
                for parent in parents:
                    if pool_cnt[i] > np.amin(pool_cnt):
                        #remove ONE pool in this path
                        self.remove_pool(parent)
                        is_updated = True
                    i+=1
                topo_cnt[update_node] = np.amin(pool_cnt)#update bfs_cnt
                if is_updated:
                    return self.update_pool(after_add)
            #add pool
            else:
                parents = list(self._graph.predecessors(update_node))
                if len(parents)==0:
                    continue
                pool_cnt = np.zeros(len(parents))
                i = 0
                for parent in parents:
                    if self._graph.node[parent]['type'] == LAYERS.maxpool or self._graph.node[parent]['type'] == LAYERS.avgpool:
                        pool_cnt[i] = topo_cnt[parent]+1
                    else:
                        pool_cnt[i] = topo_cnt[parent]
                    i+=1
                #add pool recursively
                i = 0
                for parent in parents:
                    if pool_cnt[i] < np.amax(pool_cnt):
                        #directly add ONE pool in this path (parent, pool) (pool, update_node)
                        new_pool = self.add_node(LAYERS.maxpool)
                        self.add_edge(parent, new_pool)
                        self.add_edge(new_pool, update_node)
                        self.remove_edge(parent, update_node)
                        is_updated = True
                    i+=1
                topo_cnt[update_node] = np.amax(pool_cnt)#update bfs_cnt
                if is_updated:
                    return self.update_pool(after_add)

    def get_num_layers(self):
        return self.layer_count

    def get_total_mass(self):
        return self.total_lm


    def mut_dup_path(self):
        stop_count = random.randint(1, self.layer_count-1)
        while True:
            pick = random.randint(0, self.layer_count-1)#u1
            nodes  =list(self.get_nodes())
            node = nodes[pick]
            if not (self._graph.node[node]['type'] == LAYERS.fc or self._graph.node[node]['type'] == LAYERS.ip or self._graph.node[node]['type'] == LAYERS.op or self._graph.node[node]['type'] == LAYERS.softmax):
                break
        head = node
        end = node
        new_head = node
        new_end = node
        for _ in range(stop_count):
            #reach the end
            if self._graph.node[head]['type'] == LAYERS.fc:
                break
            childs = list(self._graph.successors(head))
            pick_child = childs[random.randint(0, len(childs)-1)]
            #copy
            new_end = self.add_node(self._graph.node[pick_child]['type'], self._graph.node[pick_child]['num_of_filters'], self._graph.node[pick_child]['stride'])
            self.add_edge(new_head, new_end)
            #update and store
            new_head = new_end
            head = pick_child
        #converge
        self.add_edge(new_head, next(self._graph.successors(head)))

    def mut_remove_layer(self):
        is_pool = False
        nodes = list(self.get_nodes())
        pick_node = False
        for _ in range(10):
            pick = random.randint(0, self.layer_count-1)
            node = nodes[pick]
            #print('pick_trial: ', self._graph.node[node])#dict
            if self._graph.node[node]['type'] == LAYERS.maxpool or self._graph.node[node]['type'] == LAYERS.avgpool:
                is_pool = True
                pick_node = True
                break
            if self._graph.node[node]['type'] == LAYERS.conv3 or self._graph.node[node]['type'] == LAYERS.conv5 or self._graph.node[node]['type'] == LAYERS.conv7:
                #continue
                pick_node = True
                break
        print('removing: ', self._graph.node[node])#dict
        if not pick_node:
            return
        for parent in self._graph.predecessors(node):
            if sum(1 for _ in self._graph.successors(parent)) == 1:
                #only child for this parent
                self.add_edge(parent, next(self._graph.successors(node)))#connect parent u with del's child
        for child in self._graph.successors(node):
            if sum(1 for _ in self._graph.predecessors(child)) == 1:
                #only parent for this child
                self.add_edge(next(self._graph.predecessors(node)), child)#connect child u with del's parent
        self._graph.remove_node(node)
        self.layer_count -= 1
        #remove pool requires update for other paths
        if is_pool:
            self.update_pool(after_add = False)

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
            labels[n] = str(n) + '*' + str(t['type']) + ' ' + str(t['stride']) + ' ' + str(t['num_of_filters'])
        nx.draw_kamada_kawai(self._graph, labels=labels, node_size=node_size)
        # plt.show()

    def mut_skip(self):
        def random_pick():
            A = random.randint(1, self.layer_count-1)
            while self.get_node_attr(A) not in self.process_layers:
                A = random.randint(1, self.layer_count-1)
            B = random.randint(1, self.layer_count-1)
            while self.get_node_attr(B) not in self.process_layers or B == A:
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
        if i == 20: return
        '''
        path = nx.shortest_path(self._graph, source=nodes[0], target=nodes[1])
        pool_counter = 0
        for node in path[1:-1]:
            if self.get_node_attr(node) in self.pool_layers:
                pool_counter += 1
        while pool_counter != 0:
            nodes[0] = self.append(LAYERS.avgpool, append_to=nodes[0])
            pool_counter -= 1
        '''
        self.add_edge(nodes[0], nodes[1])
        self.update_pool(after_add=True)


    def mut_swap_label(self):
        '''
        Can I pick softmax or change to softmax?
        '''
        node = random.randint(1, self.layer_count-1)
        while self.get_node_attr(node) not in self.process_layers:
            node = random.randint(1, self.layer_count-1)
        layers_list = list(LAYERS)
        type = random.choice(layers_list)
        while type not in self.process_layers:
            type = random.choice(layers_list)
        if type in [*self.conv_layers, LAYERS.fc]:
            num_of_filters = np.random.choice([64, 128, 256, 512], 1, p=[0.4, 0.3, 0.2, 0.1])[0]
            stride = random.choice([1, 2]) if type != LAYERS.fc else 2
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
        layers_list = list(LAYERS)
        type = random.choice(layers_list)
        while type not in self.process_layers:
            type = random.choice(layers_list)
        if type in [*self.conv_layers, LAYERS.fc]:
            num_of_filters = int((self.get_node_attr(
                edge[0], 'num_of_filters') + self.get_node_attr(edge[1], 'num_of_filters')) / 2)
            if num_of_filters % 2: num_of_filters += 1
            if num_of_filters < 16:
                num_of_filters = np.random.choice([64, 128, 256, 512], 1, p=[0.4, 0.3, 0.2, 0.1])[0]
            stride = random.choice([1, 2]) if type != LAYERS.fc else 2
        else:
            num_of_filters = 1
            stride = 2
        self._graph.remove_edge(*edge)
        new_node = self.append(type, num_of_filters, stride, edge[0])
        self.add_edge(new_node, edge[1])

    def mut_step(self):
        mut_op = random.choice([self.mut_dup_path, self.mut_remove_layer,
            self.mut_dec_single, self.mut_inc_single,
            self.mut_swap_label, self.mut_wedge_layer,
            self.mut_inc_en_masse, self.mut_dec_en_masse,
            self.mut_skip])
        print(mut_op.__name__)
        mut_op()

    def mutate(self):
        num_of_steps = np.random.choice([1, 2, 3, 4, 5], 1, p=[0.5, 0.25, 0.125, 0.075, 0.05])[0]
        print('num_step: ', num_of_steps)
        for i in range(num_of_steps):
            self.mut_step()
        self.finish()

    def copy(self):
        global layer_graph_count, layer_graph_table
        G = copy.deepcopy(self)
        layer_graph_table.append(G)
        G.id = layer_graph_count
        G.finish()
        layer_graph_count += 1
        return G
