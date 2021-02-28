import numpy as np
from scipy.linalg import eig
from sklearn.metrics.pairwise import rbf_kernel

# laplacian
def laplacian(A):
    """
    :param A: M x M adjacency matrix
    :return: L: laplacian D: degree
    """
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    return L, D

# laplacian embeddings
# http://cs.uchicago.edu/~niyogi/papersps/BNnips01.ps
# http://people.cs.uchicago.edu/~niyogi/papersps/eigconv_Feb15.pdf
def laplacian_eigenmap(A):
    """
    :param A: M x M adjacency matrix
    :return: e1: eigen values v1: eigen vectors
    """
    L, D = laplacian(A)
    e, v = eig(L, D)
    e1 = e[np.argsort(e)][1:].real
    v1 = v[:, np.argsort(e)][:, 1:].real
    return e1, v1

# locality preserving projections
# https://papers.nips.cc/paper/2359-locality-preserving-projections.pdf
def lpp(X, A):
    """
    :param X: N x M feature matrix
    :param A: M x M adjacency matrix
    :return: e1: eigen, v1: eigen vectors, X projection
    """
    L, D = laplacian(A)
    l = np.einsum('ij, jk, lk', X, L, X)
    d = np.einsum('ij, jk, lk', X, D, X)
    e, v = eig(l, d)
    e1 = e[np.argsort(e)].real
    v1 = v[:, np.argsort(e)].real
    return e1, v1, np.dot(X.T, v1)

# locality preserving projections with kernel
# https://papers.nips.cc/paper/2359-locality-preserving-projections.pdf
def lpp_kernel(X, A, kernel_func=rbf_kernel):
    """
    :param X: N x M feature matrix
    :param A: M x M adjacency matrix
    :param kernel_func: kernel function
    :return: e1, v1, kernel, kernel projection
    """
    L, D = laplacian(A)
    K = kernel_func(X.T)
    l = np.einsum('ij, jk, lk', K, L, K)
    d = np.einsum('ij, jk, lk', K, D, K)
    e, v = eig(l, d)
    e1 = e[np.argsort(e)].real
    v1 = v[:, np.argsort(e)].real
    return e1, v1, K, np.dot(K, v1)