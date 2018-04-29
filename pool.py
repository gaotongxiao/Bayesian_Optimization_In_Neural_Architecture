import random
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
        self.timestep = 0
        self.cur_min = 1000

    def elim(self):
        for i, _ in self.models:
            i.elim_LAYERS()

    def rec(self):
        for idx, m in enumerate(self.models):
            m[0].rec_LAYERS()
            m[0].renew_id()
            if m[1] is None:
                m[0].show_graph()
                plt.show()
                ip = input("Input the accuracy, left empty if none:\n")
                if ip != '':
                    self.models[idx][1] = -math.log(float(ip))
                    self.cur_min = min(self.cur_min, self.models[idx][1])
    
    def append(self, graph, acc=None):
        if not acc is None:
            acc = -math.log(acc)
            self.cur_min = min(self.cur_min, acc)
        self.models.append([graph, acc])

    def get_training_data(self):
        X = []
        Y = []
        for m in self.models:
            if not m[1] is None:
               X.append(m[0])
               Y.append(m[1]) 
        return X, Y
    
    def get_prob(self, acc):
        std = np.std(acc)
        return np.exp(acc / std)
    
    def get_pred_data(self):
        x = []
        for m in self.models:
            if m[1] is None:
               x.append(m[0])
        return x

    def get_layer_graph(self, graph_idx):
        return self.models[graph_idx][0]

    def get_layer_graph_acc(self, graph_idx):
        return self.models[graph_idx][1]

    def mutate_layer_graph(self, graph_idx):
        mut_graph = copy.deepcopy(self.get_layer_graph(graph_idx))
        mut_graph.mutate()
        self.models.append([mut_graph, None])

    def mutate(self):
        self.timestep += 1
        # N_mut = math.ceil(math.sqrt(self.timestep))
        N_mut = 2 * self.timestep
        n = math.floor(math.sqrt(self.timestep))
        res_graph = []
        mut_pools = []
        X, Y = self.get_training_data()
        netModel = NetModel(X)
        netModel.mcmc(Y)
        for m in random.choices(X, k=N_mut, weights=self.get_prob(Y)):
            new_m = m.copy()
            new_m.mutate()
            new_m_acq = netModel.marginal_acquisition_func(new_m, Y, self.cur_min, sample_time=1000)
            mut_pools.append([new_m, new_m_acq])
        print(list(zip(*mut_pools))[1])
        for mp in sorted(mut_pools, key=lambda x: x[1], reverse=True)[:n]:
            self.append(mp[0])
            mp[0].show_graph()
        plt.show()
        

def write(pl_obj, path='pool'):
    pl_obj.elim()
    pickle.dump(pl_obj, open(path, 'wb'))

def read(path='pool'):
    lg.clear_layers()
    pl_obj = pickle.load(open(path, 'rb'))
    pl_obj.rec()
    return pl_obj

if __name__ == '__main__':
    pooln = input("Enter the pool number:")
    P = read('models/pool' + pooln)
    P.mutate()
    write(P, 'models/pool' + str(int(pooln) + 1))
    # exit()
    # P.mutate_layer_graph(0)

    # print(netModel.K([P.get_layer_graph(1), P.get_layer_graph(0)], [P.get_layer_graph(1), P.get_layer_graph(0)]))
    # gt1 = P.get_layer_graph(3)
    # gt2 = P.get_layer_graph(4)
    # print(netModel.mean_cond([P.get_layer_graph(5)], [gt1,gt2], [0.1,0.2]))
    # print(netKernel.K([P.get_layer_graph(1), P.get_layer_graph(0)], [P.get_layer_graph(1), P.get_layer_graph(0)]))
    X, Y = P.get_training_data()
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
        print(netModel.marginal_acquisition_func(mut_pool, Y, min(Y), sample_time=1000))
