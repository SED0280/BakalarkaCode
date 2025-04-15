import numpy as np
import integrations as integ


def get_eigenfuntions(x, poly_count, node_count):
    def legnorm(x): return integ.legendre_normal(x, poly_count)
    nodes, weights = integ.nodes_weights(node_count)

    nodesx1 = np.reshape(nodes, (-1, 1, 1, 1))
    nodesy1 = np.reshape(nodes, (1, -1, 1, 1))
    nodesx2 = np.reshape(nodes, (1, 1, -1, 1))
    nodesy2 = np.reshape(nodes, (1, 1, 1, -1))
    values = legnorm(nodes)

    # print(cov(nodesx1, nodesx2, nodesy1, nodesy2).shape)
    # print(values.shape)
    dist_sq = (nodesx1 - nodesx2)**2 + (nodesy1 - nodesy2)**2
    C_tensor = np.exp(-np.sqrt(dist_sq))

    PW = (values * weights)
    A_tensor = np.einsum('kp,lq,pqrs,ir,js->klij', PW, PW,
                         C_tensor, PW, PW, optimize=True)
    A_tensor.shape
    A_matrix = np.reshape(A_tensor, ((poly_count+1)**2, (poly_count+1)**2))
    eigenvalues, eigenvectors = np.linalg.eigh(A_matrix)
    values = legnorm(x)

    # Perform broadcasted multiplication
    V1 = values[:, np.newaxis, np.newaxis, :]  # Shape (L, 1, 1, N)
    V2 = values[np.newaxis, :, :, np.newaxis]  # Shape (1, L, N, 1)
    V = V1 * V2
    V = np.reshape(V, (len(V)**2, len(x), len(x)))
    eigenvectors_functions = (np.einsum(
        'ijk,in->njk', V, eigenvectors, optimize=True).T * np.sqrt(
        np.maximum(eigenvalues, 0))).T
    return eigenvectors_functions


def sample_normal(eigenfunctions):
    n = eigenfunctions.shape[0]
    realisation = np.random.normal(size=n)
    realisation = eigenfunctions.T @ realisation
    return realisation
