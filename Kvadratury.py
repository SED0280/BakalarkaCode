import numpy as np
import scipy.integrate as integ

def w(xzeros, j):
    prod = np.polynomial.Polynomial.fromroots([xzeros[i] for i in range(len(xzeros)) if i!=j])
    bottom = prod(xzeros[j])
    top = prod.integ()
    return top/bottom

def getWeights(xzeros):
    return [w(xzeros, i) for i in range(len(xzeros))]


ws = getWeights([1, 2, 3])
for i in ws:
    print(i)
