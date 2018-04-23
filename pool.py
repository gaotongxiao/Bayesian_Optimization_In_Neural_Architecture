from layer_graph import LAYERS, Layer_graph
import layer_graph as lg
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from model_util import NetModel
import pickle

class Pool(object):
    def __init__(self):
        self.models = []

    def elim(self):
        for i, _ in self.models:
            i.elim_LAYERS()

    def rec(self):
        for i, _ in self.models:
            i.rec_LAYERS()
            i.renew_id()
    
    def append(self, graph, acc=None):
        if not acc is None:
            acc = -math.log(acc)
        self.models.append((graph, acc))
        
    def get_layer_graph(self, graph_idx):
        return self.models[graph_idx][0]

    def get_layer_graph_acc(self, graph_idx):
        return self.models[graph_idx][1]

    def mutate_layer_graph(self, graph_idx):
        mut_graph = copy.deepcopy(self.get_layer_graph(graph_idx))
        mut_graph.mutate()
        self.models.append((mut_graph, None))

def write(pl_obj, path='pool'):
    pl_obj.elim()
    pickle.dump(pl_obj, open(path, 'wb'))

def read(path='pool'):
    lg.clear_layers()
    pl_obj = pickle.load(open(path, 'rb'))
    pl_obj.rec()
    return pl_obj

if __name__ == '__main__':
    P = read('models/pool')
    # P.mutate_layer_graph(0)
    mut_pool = P.get_layer_graph(1).copy()
    mut_pool.mut_skip()
    mut_pool.mut_skip()
    mut_pool.mut_skip()
    mut_pool.finish()

    # print(netModel.K([P.get_layer_graph(1), P.get_layer_graph(0)], [P.get_layer_graph(1), P.get_layer_graph(0)]))
    # gt1 = P.get_layer_graph(3)
    # gt2 = P.get_layer_graph(4)
    # print(netModel.mean_cond([P.get_layer_graph(5)], [gt1,gt2], [0.1,0.2]))
    # print(netKernel.K([P.get_layer_graph(1), P.get_layer_graph(0)], [P.get_layer_graph(1), P.get_layer_graph(0)]))
    X = list(list(zip(*(P.models)))[0])
    Y = list(list(zip(*(P.models)))[1])
    netModel = NetModel(X)
    # for i in range(100):
    #     print(netModel.post_K(mut_pool, mut_pool, X))
    #     mut_pool.mutate()
    # print(netModel.post_K(mut_pool, mut_pool, X))
    # print(netModel.acquisition_func(mut_pool, X, Y, max(Y)))
    # print(netModel.post_dist(mut_pool, 0.9, X, Y))
    netModel.mcmc(Y)
    for x in range(10):
        pass
        print(netModel.marginal_acquisition_func(mut_pool, Y, min(Y), sample_time=100))
