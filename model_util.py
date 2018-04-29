from distance import get_distance
import numpy as np
import emcee
from scipy.stats import norm, multivariate_normal
import random
import matplotlib.pyplot as plt
import math

class NetModel():
    def __init__(self, X):
        self.alpha = 0.5
        self.alpha_bar = 0.5
        # self.betas = [0.1, 0.2, 0.3, 0.4]
        self.betas = [0.2]
        # self.beta_bars = [0.1, 0.2, 0.3, 0.4]
        self.beta_bars = [0.2]
        # self.v_str = [0.1, 0.2, 0.4, 0.8]
        self.v_str = [0.1]
        self.num_beta = 1
        self.num_of_paras = 4
        self.burn_in_steps = 100
        self.production_chain_steps = 1000
        self.distance_matrix_list = [] # tuple
        self.X = X
        self.X_dim = len(X)
        self.init_distance_matrix(X)
        
    def init_distance_matrix(self, X1, X2=None):
        if X2 == None:
            X2 = X1
        if not isinstance(X1, list):
            X1 = [X1]
        if not isinstance(X2, list):
            X2 = [X2]
        n_1 = len(X1)
        n_2 = len(X2)
        distance_matrix = np.zeros((n_1, n_2))
        distance_bar_matrix = np.zeros((n_1, n_2))
        for k in range(self.num_beta):
            for i in range(n_1):
                for j in range(n_2):
                        d, d_bar = get_distance(X1[i], X2[j], self.v_str[k])
                        distance_matrix[i, j] = d
                        distance_bar_matrix[i, j] = d_bar
            self.distance_matrix_list.append((distance_matrix, distance_bar_matrix))

    def mu(self, X, constant=0):
        if X == None:
            return constant*np.ones(self.X_dim)
        return constant*np.ones_like(X)

    def K(self, X1, X2):
        def K_single(x1, x2):
            first_term = self.alpha*np.exp(sum([-self.betas[i]*get_distance(x1, x2,self.v_str[i])[0] for i in range(self.num_beta)]))
            second_term = self.alpha_bar*np.exp(sum([-self.beta_bars[i]*(get_distance(x1, x2, self.v_str[i])[1])**2 for i in range(self.num_beta)]))
            return first_term + second_term
        if X1 == None:
            X1 = self.X
        if X2 == None:
            X2 = self.X
        if not isinstance(X1, list):
            X1 = [X1]
        if not isinstance(X2, list):
            X2 = [X2]
        n_1 = len(X1)
        n_2 = len(X2)
        ret = np.zeros((n_1, n_2))
        for i in range(n_1):
            for j in range(n_2):
                ret[i, j] = K_single(X1[i], X2[j])
        return ret

    def K_XX(self, X=None):

        if not X == None:
            return self.K(X, X)

        def K_XX_single(xi, xj):
            first_term = self.alpha*np.exp(sum([-self.betas[i]*self.distance_matrix_list[i][0][xi, xj] for i in range(self.num_beta)]))
            second_term = self.alpha_bar*np.exp(sum([-self.beta_bars[i]*self.distance_matrix_list[i][1][xi, xj]**2 for i in range(self.num_beta)]))
            return first_term + second_term

        ret = np.zeros_like(self.distance_matrix_list[0][0])
        n = np.shape(ret)[0]
        for i in range(n):
            for j in range(n):
                ret[i, j] = K_XX_single(i, j)
        # w, v = np.linalg.eig(ret)
        # w = np.absolute(w)
        # return w * v# + 1e-10
        return ret

    def post_K(self, x1, x2, X=None):
        res = self.K(x1, x2) - self.K(x1, X).dot(np.linalg.inv(self.K_XX(X))).dot(self.K(X, x2))
        # return res
        w, v = np.linalg.eig(res)
        w = np.absolute(w)
        return w * v + 1e-10

    
    def post_mu(self, x, Y, X=None):
        return self.mu(x) + self.K(x, X).dot(np.linalg.inv(self.K_XX(X)).dot((np.array(Y).T - self.mu(x))))

    def acquisition_func(self, x, Y, cur_min, X=None):
        mu_x = self.post_mu(x, Y, X)
        K_xx = self.post_K(x, x, X)
        return np.squeeze((cur_min - mu_x)*norm.cdf(cur_min, mu_x, np.sqrt(K_xx)) + K_xx*norm.pdf(cur_min, mu_x, np.sqrt(K_xx)))

    def marginal_acquisition_func(self, x, Y, cur_min, X=None, sample_time=10):
        '''
        Assume that mcmc has been run before this function
        '''
        if self.sampler is None:
            print("Make sure run mcmc before calling this function!")
            return
        res = 0
        for _ in range(sample_time):
            while True:
                f = random.randint(0, self.num_of_paras * 2 * (self.production_chain_steps + self.burn_in_steps)-1)
                while self.sampler.flatlnprobability[f] == -np.inf:
                    f = random.randint(0, self.num_of_paras * 2 * (self.production_chain_steps + self.burn_in_steps)-1)
                p = self.sampler.flatchain[f]
                self.alpha = p[0]/(p[0]+p[1])
                self.alpha_bar = p[1]/(p[0]+p[1])
                self.betas = [p[2]]
                self.beta_bars = [p[3]]
                func_value = self.acquisition_func(x, Y, cur_min, X)
                if math.isnan(func_value):
                    continue
                res += func_value
                break
        return res / sample_time

    def mcmc(self, Y, X=None, burn_in_steps=100, production_chain_steps=1000):
        def lnprob(p):
            if np.any((p < 0) + (p > 10)):
                return -np.inf
            self.alpha = p[0]/(p[0]+p[1])
            self.alpha_bar = p[1]/(p[0]+p[1])
            self.betas = [p[2]]
            self.beta_bars = [p[3]]
            return np.log(self.data_likelihood(Y, X))
        
        self.burn_in_steps = burn_in_steps
        self.production_chain_steps = production_chain_steps
        nwalkers, ndim = self.num_of_paras * 2, self.num_of_paras
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

        # Initialize the walkers.
        p0 = 2.5 * np.ones((4)) + 5e-2 * np.random.randn(nwalkers, ndim)

        print("Running burn-in")
        p0, _, _ = self.sampler.run_mcmc(p0, self.burn_in_steps)

        print("Running production chain")
        self.sampler.run_mcmc(p0, self.production_chain_steps)
        # x = range(0, nwalkers * (self.burn_in_steps + self.production_chain_steps))
        # for i in range(self.num_of_paras):
        #     plt.figure(i)
        #     plt.plot(x, self.sampler.flatchain[:, i])
        # plt.figure(self.num_of_paras + 1)
        # plt.plot(x, np.exp(self.sampler.flatlnprobability))
        # plt.show()

    def data_likelihood(self, Y, X=None):
        a = self.K_XX(X)
        try:
            ret = multivariate_normal.pdf(Y, self.mu(X), a)
        except:
            print(a)
            exit()
        return ret

    def post_dist_pdf(self, X_star, Y_star, Y, X=None):
        return multivariate_normal.pdf(Y_star, self.post_mu(X_star, Y, X), self.post_K(X_star, X_star, X))
    
