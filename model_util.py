from distance import get_distance
import numpy as np
from scipy.stats import norm, multivariate_normal

class NetModel():
    def __init__(self):
        self.alpha = 0.1
        self.alpha_bar = 0.1
        self.betas = [0.1, 0.2, 0.3, 0.4]
        self.beta_bars = [0.1, 0.2, 0.3, 0.4]
        self.v_str = [0.1, 0.2, 0.4, 0.8]

    def mu(self, X, constant=0):
        return constant*np.ones_like(X)

    def K(self, X1, X2=None):
        def K_single(x1, x2):
            first_term = self.alpha*np.exp(sum([-self.betas[i]*get_distance(x1, x2,self.v_str[i])[0] for i in range(4)]))
            second_term = self.alpha_bar*np.exp(sum([-self.beta_bars[i]*(get_distance(x1, x2, self.v_str[i])[1])**2 for i in range(4)]))
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
        return self.mu(x) + self.K(x, X).dot(np.linalg.inv(self.K(X, X)).dot((np.array(Y).T - self.mu(x))))

    def acquisition_func(self, x, X, Y, cur_max):
        mu_x = self.post_mu(x, X, Y)
        K_xx = self.post_K(x, x, X)
        return (cur_max - mu_x)*norm.cdf(cur_max, mu_x, np.sqrt(K_xx)) + K_xx*norm.pdf(cur_max, mu_x, np.sqrt(K_xx))

    def data_likelihood(self, X, Y):
        return multivariate_normal.pdf(Y, self.mu(X), self.K(X, X))

    def post_dist(self, X_star, Y_star, X, Y):
        return multivariate_normal.pdf(Y_star, self.post_mu(X_star, X, Y), self.post_K(X_star, X_star, X))
