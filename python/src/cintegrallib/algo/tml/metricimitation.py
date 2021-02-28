from sklearn.manifold import _locally_linear as ll
from scipy.linalg import eig
import numpy as np

# cross feature domains metrics imitation
# http://people.ee.ethz.ch/~daid/publications/MetricImitation_CVPR15.pdf
# X: source features matrix: N samples x M features
# Y: target features matrix" N samples x Q features
# in practice Q << M
def projection_matrix(X, Y, n_neighbors=2):
    assert(Y.shape[0] > Y.shape[1]) # samples dim > features dim

    W = ll.barycenter_kneighbors_graph(X, n_neighbors)
    M = (W.T * W - W.T - W).toarray()
    M.flat[::M.shape[0] + 1] += 1

    gramYM = np.einsum('ij, ii, ik', Y, M, Y)
    gramY = np.einsum('ij, ik', Y, Y)
    e, v = eig(gramYM, gramY)
    e1 = e[np.argsort(e)].real
    v1 = v[:, np.argsort(e)].real
    return e1, v1
