import numpy as np
import matplotlib.pyplot as plt

def BinArray(arr: np.ndarray, threshold: float) -> np.ndarray:

    return (arr > threshold).astype(int)

def Normalizer(A, eps_ = 0.0000000000001):

    return (A - np.mean(A)) / (np.std(A) + eps_)

def SimpleSparsityGraphLearning(W, ft = 0.1, st = 20, stp = 0.1, lambda_ = 0.1, plotplot = False, ax = None):

    OpTerm = []
    Thrs = 1 / np.arange(ft, st, stp)

    for Thr in Thrs:

        A = BinArray(W, Thr)

        OpTerm.append(np.linalg.norm(Normalizer(A) - Normalizer(W)) + lambda_ * np.linalg.norm(A))

    i_opt = np.argmin(OpTerm)
    OptThr = Thrs[i_opt]

    if plotplot:

        if ax == None:

            fig, ax = plt.subplots(1, 1)


        ax.plot(Thrs, OpTerm)
        ax.axvline(OptThr, color = 'r')
        plt.show()

    return OptThr, BinArray(W, OptThr)

def _load_coords(dir = r'D:\AIRLab_Research\To Write the Paper\Paper D\ch_coords.npy'):

    FCoords = np.load(dir, allow_pickle =True).item()

    ValidChannels = [key for key in FCoords.keys()]
    CoordMat = np.array([FCoords[ValidChannel] for ValidChannel in ValidChannels])

    return ValidChannels, CoordMat

def EucDist(Ap, Bp):

    return np.sqrt(np.sum((Ap - Bp) ** 2))

def DistWeightMat(Locs):

    Nn, _ = Locs.shape

    RetGraph = np.zeros((Nn, Nn))

    for i in range(Nn):

        for j in range(i):

            Dist = EucDist(Locs[i], Locs[j])

            RetGraph[i, j] = 1 / (Dist + 0.0000000001)
            RetGraph[j, i] = RetGraph[i, j]

    return RetGraph

def D_mat(W):

    return np.diag(np.sum(W, axis = 0))

def L_mat(W, normalized=False):
    """
    Compute Laplacian or normalized Laplacian matrix.

    Parameters
    ----------
    W : np.ndarray
        Adjacency matrix (N×N).
    normalized : bool, default=False
        If True, return symmetric normalized Laplacian.

    Returns
    -------
    L : np.ndarray
        Laplacian matrix (normalized if specified).
    """
    D = D_mat(W)

    if normalized:
        # Avoid division by zero for isolated nodes
        with np.errstate(divide='ignore'):
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        L_norm = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        return L_norm
    else:
        return D - W