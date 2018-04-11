import ot
import numpy as np

a = np.array([1, 1])
b = [1, 1]
M = [[0, 1], [1, 0]]
print(ot.emd(a, b, M))
