import numpy as np
import scipy.integrate as integ

def weight(x, xzeros, j):
    prod = np.polynomial.Polynomial.fromroots([xzeros[i] for i in range(len(xzeros)) if i!=j])
    bottom = prod(xzeros[j])
    print(integ(prod))



weight(7.4, [1, 2, 3], 2)
