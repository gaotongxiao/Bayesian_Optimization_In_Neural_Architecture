from distance import get_distance
import numpy as np
import emcee
from scipy.stats import norm, multivariate_normal

class NetModel():
    def __init__(self):
        self.alpha = 0.1
        self.alpha_bar = 0.1
        # self.betas = [0.1, 0.2, 0.3, 0.4]
        self.betas = [0.1]
        # self.beta_bars = [0.1, 0.2, 0.3, 0.4]
        self.beta_bars = [0.1]
        # self.v_str = [0.1, 0.2, 0.4, 0.8]
        self.v_str = [0.1]
        self.num_beta = 1
        self.num_of_paras = 4
        
    def K(self, X1, X2=None):
        def K_single(x1, x2):
            first_term = self.alpha*np.exp(sum([-self.betas[i]*get_distance(x1, x2,self.v_str[i])[0] for i in range(self.num_beta)]))
            second_term = self.alpha_bar*np.exp(sum([-self.beta_bars[i]*get_distance(x1, x2, self.v_str[i])[1] for i in range(self.num_beta)]))
            return first_term + second_term
        if not X2:
            X2 = X1
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

    def post_K(self, x1, x2, X):
        return self.K(x1, x2) - self.K(x1, X).dot(np.linalg.inv(self.K(X, X))).dot(self.K(X, x2))

    
    def post_mu(self, x, X, Y):
        return self.mu + self.K(x, X).dot(np.linalg.inv(self.K(X, X)).dot((np.array(Y).T - self.mu)))

    def acquisition_func(self, x, X, Y, cur_max):
        mu_x = self.post_mu(x, X, Y)
        K_xx = self.post_K(x, x, X)
        return (cur_max - mu_x)*norm.cdf(cur_max, mu_x, np.sqrt(K_xx)) + K_xx*norm.pdf(cur_max, mu_x, np.sqrt(K_xx))


    def post_dist(self, X_star, Y_star, X, Y):
        return multivariate_normal.pdf(Y_star, self.post_mu(X_star, X, Y), self.post_K(X_star, X_star, X))

    def mcmc(self):
        def lnprob(p):
            if np.any(p < 0 + p > 1):
                return -np.inf
            self.alpha = p[0]
            self.alpha_bar = p[1]
            self.betas = [p[2]]
            self.beta_bars = [p[3]]
            return self.post_dist(....)
        nwalkers, ndim = 36, self.num_of_paras
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

        # Initialize the walkers.
        p0 = 0.5 * np.ones((4)) + 1e-4 * np.random.randn(nwalkers, ndim)

        print("Running burn-in")
        p0, _, _ = sampler.run_mcmc(p0, 200)

        print("Running production chain")
        sampler.run_mcmc(p0, 200);
        print(sampler.chain)