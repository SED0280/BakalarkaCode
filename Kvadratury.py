import numpy as np
import scipy.integrate as integ
import math

def w(xzeros, j):
    prod = np.polynomial.Polynomial.fromroots([xzeros[i] for i in range(len(xzeros)) if i!=j])
    bottom = prod(xzeros[j])
    top = prod.integ()(1)-prod.integ()(-1)
    return top/bottom

def getWeights(xzeros):
    return [w(xzeros, i) for i in range(len(xzeros))]

def generateLegendre(x, n):
    alphak = 0
    ret = [np.zeros(len(x)), np.ones(len(x))]
    for k in range(2,n+1):
        betak = 1/(4-(k-1)**(-2))
        pi = (x-alphak)*ret[-1]-betak*ret[-2]
        ret.append(pi)
    return np.array(ret[1:])

def generateLegendrePoly1(n):
    alphak = 0
    ret = [lambda x: 0, lambda x: 1]
    for k in range(2,n+1):
        betak = 1/(4-(k-1)**(-2)) # nebo mam pocitat pomoci integralu/G. kvadratur?
        pi = (lambda x:(x-alphak)*ret[-1](x)-betak*ret[-2](x))
        ret.append(pi)
    return np.array(ret[1:])

#print(generateLegendrePoly(2)[1](5))
def generateLegendrePoly(n):
    ret = [lambda x: 0, lambda x: 1]
    for k in range(2,n+1):
        betak = 1/(4-(k-1)**(-2))
        def pi(x, Pk_minus_1=ret[-1], Pk_minus_2=ret[-2], betak = betak):
            alphak = 0 
            #print("   ", betak)
            return (x - alphak) * Pk_minus_1(x) - betak * Pk_minus_2(x)
        #print(pk(1),"\n-------------------\n")
        ret.append(pi)
    return np.array(ret[1:])

def generateNormalLegendrePoly(n):
    ret = [lambda x: 0, lambda x: 1]
    
    for k in range(2,n+1):
        betak_minus_one = 1/(4-(k-1)**(-2))
        betak = 1/(4-k**(-2))
        def pi(x, Pi_minus_1=ret[-1], Pi_minus_2=ret[-2], betak = betak, betak_minus_one = betak_minus_one):
            alphak = 0 
            #print("   ", betak)
            return ((x - alphak) * Pi_minus_1(x) - math.sqrt(betak_minus_one) * Pi_minus_2(x))/math.sqrt(betak)
        #print(pk(1),"\n-------------------\n")
        #print(lastbetak,"   ",betak,"\n")
        ret.append(pi)
    return np.array(ret[1:])

n=10
beta=[2]
alpha=[0]
for k in range(2,n):
    beta.append(1/(4-(k-1)**(-2)))
    alpha.append(0)
alpha.append(0)

import numpy as np

# Example of constructing a Jacobi matrix (for simplicity, this is a dummy example)
J_n = np.array([[2, 1, 0, 0],
                [1, 2, 1, 0],
                [0, 1, 2, 1],
                [0, 0, 1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(J_n)

# Sort eigenvalues (roots) and eigenvectors
sorted_indices = np.argsort(eigenvalues)
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

#print("Eigenvalues (roots):", eigenvalues_sorted[0])
#print("Eigenvectors:", eigenvectors_sorted[:,0])

#print("-------")
#print("Eigenvalues (roots):", eigenvalues[2])
#print("Eigenvectors:", eigenvectors[:,2])
#print(np.dot(J_n,eigenvectors[2]))
#print(np.dot(eigenvalues[2],eigenvectors[2]))
