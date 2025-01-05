import numpy as np
import scipy.integrate as integ

def w(xzeros, j):
    prod = np.polynomial.Polynomial.fromroots([xzeros[i] for i in range(len(xzeros)) if i!=j])
    bottom = prod(xzeros[j])
    top = prod.integ()
    return top/bottom

def getWeights(xzeros):
    return [w(xzeros, i) for i in range(len(xzeros))]

def generateLegendrePolynomials(x, n):
    alphak = 0
    ret = [np.zeros(len(x)), np.ones(len(x))]
    for k in range(2,n+1):
        betak = 1/(4-k**(-2))
        pi = (x-alphak)*ret[-1]-betak*ret[-2]
        ret.append(pi)
    return np.array(ret[1:])


