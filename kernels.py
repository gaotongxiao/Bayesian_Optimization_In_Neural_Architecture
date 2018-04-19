from distance import get_distance
import numpy as np

class NetKernel():
    def __init__(self):
        self.alpha = 0.1
        self.alpha_bar = 0.1
        self.betas = [0.1, 0.2, 0.3, 0.4]
        self.beta_bars = [0.1, 0.2, 0.3, 0.4]
        self.v_str = [0.1, 0.2, 0.4, 0.8]

    def K(self, X1, X2=None):
        def K_single(x1, x2):
            first_term = self.alpha*np.exp(sum([-self.betas[i]*get_distance(x1, x2,self.v_str[i])[0] for i in range(4)]))
            second_term = self.alpha_bar*np.exp(sum([-self.beta_bars[i]*get_distance(x1, x2, self.v_str[i])[1] for i in range(4)]))
            return first_term + second_term
        n_1 = len(X1)
        n_2 = len(X2)
        ret = np.zeros((n_1, n_2))
        for i in range(n_1):
            for j in range(n_2):
                ret[i, j] = K_single(X1[i], X2[j])
        return ret
    
