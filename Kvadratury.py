import numpy as np
import scipy.integrate as integ
from scipy.linalg import eigh_tridiagonal
import math

def w(xzeros, j):
    prod = np.polynomial.Polynomial.fromroots([xzeros[i] for i in range(len(xzeros)) if i!=j])
    bottom = prod(xzeros[j])
    top = prod.integ()(1)-prod.integ()(-1)
    return top/bottom

def getWeights(xzeros):
    return np.array([w(xzeros, i) for i in range(len(xzeros))])

def generateLegendre(x, n):
    alphak = 0
    ret = [np.zeros(len(x)), np.ones(len(x))]
    for k in range(2,n+1):
        betak = 1/(4-(k-1)**(-2))
        pi = (x-alphak)*ret[-1]-betak*ret[-2]
        ret.append(pi)
    return np.array(ret[1:])

def generateNormalLegendre(x, n):
    ret = [np.zeros(len(x)), np.ones(len(x))]
    alphak = 0
    for k in range(2,n+1):
        betak_minus_one = 1/(4-(k-1)**(-2))
        betak = 1/(4-(k-1)**(-2))

        pi = ((x - alphak) * ret[-1] - math.sqrt(betak_minus_one) * ret[-2])/math.sqrt(betak)
        ret.append(pi)
    return np.array(ret[1:])

def generateLegendreInterval(x, n, a, b):
    alphak = 0
    ret = [np.zeros(len(x)), np.ones(len(x))]
    for k in range(2,n+1):
        betak = 1/(4-(k-1)**(-2))
        pi = ( (((a+b)/(a-b) + (2/(b-a))*x) -alphak)*ret[-1]-betak*ret[-2] )
        #pi =math.sqrt(2/(b-a))*pi
        ret.append(pi)
    return np.array(ret[1:])

def generateNormalLegendreInterval(x, n, a, b):
    ret = [np.zeros(len(x)), np.ones(len(x))]
    alphak = 0
    for k in range(2,n+1):
        betak_minus_one = 1/(4-(k-1)**(-2))
        betak = 1/(4-(k-1)**(-2))

        pi = ((((a+b)/(a-b) + 2/(b-a)*x) - alphak) * ret[-1] - math.sqrt(betak_minus_one) * ret[-2])/math.sqrt(betak)
        #pi = math.sqrt(2/(b-a))*pi
        ret.append(pi)
    return np.array(ret[1:])

"""
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
"""

def JacobiGaussQuadLegendre(f, n):
    Beta0 = 2
    main_diag = np.zeros(n)
    off_diag = np.array([ math.sqrt(1/(4- k**(-2))) for k in range(1,n)])
    
    eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

    #J_n = np.diag(main_diag)
    #J_n[1:,:len(J_n)-1]+=off_diag
    #J_n[:len(J_n)-1,1:]+=off_diag
    #print(np.dot(J_n,eigenvectors[:,0]))
    #print(np.dot(eigenvalues[0],eigenvectors[:,0]))

    lam = Beta0 * (eigenvectors[0]**2)
    #lam = getWeights(eigenvalues)
    #print(getWeights(eigenvalues))
    #print(lam)

    return sum(lam*f(eigenvalues))

def JacobiGaussQuadLegendre1(f, n):
    Beta0 = 2
    main_diag = np.zeros(n)
    off_diag = np.array([ math.sqrt(1/(4- k**(-2))) for k in range(1,n)])
    
    eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

    #J_n = np.diag(main_diag)
    #J_n[1:,:len(J_n)-1]+=off_diag
    #J_n[:len(J_n)-1,1:]+=off_diag
    #print(np.dot(J_n,eigenvectors[:,0]))
    #print(np.dot(eigenvalues[0],eigenvectors[:,0]))

    #lam = Beta0 * eigenvectors[0]
    lam = getWeights(eigenvalues)
    #print(getWeights(eigenvalues))
    #print(lam)

    return sum(lam*f(eigenvalues))



def JacobiGaussQuadLegendreInterval(f, n, a, b):
    Beta0 = 2
    main_diag = np.zeros(n)
    off_diag = np.array([ math.sqrt(1/(4- k**(-2))) for k in range(1,n)])
    
    eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

    
    eigenvalues = (a+b)/2 + (b-a)/2*eigenvalues
    lam = Beta0 * (eigenvectors[0]**2) * (b-a)/2
    #print(getWeights(eigenvalues))
    #print(lam)

    return sum(lam*f(eigenvalues))

def JacobiGaussQuadLegendreInterval1(f, n, a, b):
    Beta0 = 2
    main_diag = np.zeros(n)
    off_diag = np.array([ math.sqrt(1/(4- k**(-2))) for k in range(1,n)])
    
    eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

    
    lam = (b-a)/2*getWeights(eigenvalues)
    eigenvalues = (a+b)/2 + (b-a)/2*eigenvalues
    #print(getWeights(eigenvalues))
    #print(lam)

    return sum(lam*f(eigenvalues))

#n=100
#f = lambda x: x**2

#print("Integral of x^2 over <-1,1> = ", JacobiGaussQuadLegendre(f, n))

#J_n = np.diag(main_diag, off_diag)
#print(J_n)
#print(np.dot(J_n,eigenvectors[2]))
#print(np.dot(eigenvalues[2],eigenvectors[2]))

