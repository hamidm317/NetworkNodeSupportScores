import numpy as np
import numpy.linalg as npl
from typing import Dict, Tuple, Optional, List

# =========================================================
# Joint Time–Graph AR (JT-GAR) where temporal lag == graph power
# Model (default, identifiable):
#   x_t ≈ sum_{r=1..R} a_r * S^r * x_{t-r}
# =========================================================

# ----------------------------- Utilities -----------------------------

def jt_build_graph_powers(S: np.ndarray, R: int) -> Dict[int, np.ndarray]:
    """
    Return a dict of powers {S^r} for r=0..R, with S^0 = I.
    """
    N = S.shape[0]
    powers = {0: np.eye(N)}
    cur = np.eye(N)
    for r in range(1, R+1):
        cur = cur @ S
        # e, V = np.linalg.eig(cur)
        # V = np.matrix(V)
        # e1 = e / np.sqrt(np.sum(e ** 2))
        # cur_ = V @ np.diag(e1) @ V.H
        powers[r] = np.real(cur)
    return powers


def jt_design_matrix(
    X: np.ndarray,
    S_powers: Dict[int, np.ndarray],
    R: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build design matrix for the JT-GAR model.

    Default identifiable model:
        x_t ≈ sum_{r=1..R} a_r S^r x_{t-r}

    Returns
    -------
    Z : (N*(T-R), R)   features stacked over time and nodes
    y : (N*(T-R),)     target stacked over time and nodes
    """
    N, T = X.shape
    n_t = T - R
    if n_t <= 0:
        raise ValueError("Not enough time points for the chosen R.")

    # Target
    
    y_mat = X[:, R:]                   # shape (N, n_t)
    y = y_mat.reshape(N * n_t, order='F')

    # Features: for each r=1..R, S^r x_{t-r}
    Z = np.zeros((N * n_t, R), dtype=X.dtype)
    for r in range(1, R+1):
        block = S_powers[r] @ X[:, R - r : T - r]   # (N, n_t)
        Z[:, r-1] = block.reshape(N * n_t, order='F')
    return Z, y


def jt_fit(
    X: np.ndarray,
    S: np.ndarray,
    R: int = 2,
    ridge_lambda: float = 1e-6,
    nonlinear_shift = False
) -> Dict[str, np.ndarray]:
    """
    Fit the JT-GAR model via ridge LS.

    Parameters
    ----------
    X : (N, T) array
    S : (N, N) shift
    R : int, maximum lag/graph power
    ridge_lambda : float, ridge regularization
    
    Returns
    -------
    model : dict with keys
        'a'           : (R,) coefficients [a_1..a_R]
        'R'           : int
        'S_powers'    : dict of {r: S^r}
        'rss'         : residual sum of squares
        'n'           : number of scalar residuals
        'train_mse'   : MSE on training
    """
    S_powers = jt_build_graph_powers(S, R)
    Z, y = jt_design_matrix(X, S_powers, R)
    # Ridge LS
    K = Z.shape[1]
    G = Z.T @ Z
    if ridge_lambda > 0:
        G = G + ridge_lambda * np.eye(K)
    rhs = Z.T @ y
    a = npl.solve(G, rhs)   # (R,)

    # Residuals
    y_hat = Z @ a
    resid = y - y_hat
    rss = float(resid.T @ resid)
    n = y.size
    mse = rss / n
    out = {
        'a': a,
        'R': R,
        'S_powers': S_powers,
        'rss': rss,
        'n': n,
        'train_mse': mse
    }

    return out


def jt_predict_one_step(X: np.ndarray, model: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-step prediction for the JT-GAR model.

    x_t ≈ sum_{r=1..R} a_r S^r x_{t-r}

    Returns
    -------
    X_hat : (N, T-R)
    resid : (N, T-R)
    """
    a = model['a']           # shape (R,)
    R = int(model['R'])
    S_powers = model['S_powers']
    N, T = X.shape
    n_t = T - R
    if n_t <= 0:
        raise ValueError("Not enough time points for prediction.")

    # Build RHS sum_r a_r S^r x_{t-r}
    rhs = np.zeros((N, n_t), dtype=X.dtype)
    for r in range(1, R+1):
        rhs += a[r-1] * (S_powers[r] @ X[:, R - r : T - r])

    X_hat = rhs

    resid = X[:, R:] - X_hat
    return X_hat, resid


def jt_compute_mse(X: np.ndarray, model: Dict[str, np.ndarray], return_residuals = False) -> float:
    """
    Mean squared error of one-step prediction under JT-GAR model.
    """
    _, resid = jt_predict_one_step(X, model)

    if return_residuals:

        return np.mean(resid ** 2, axis = 1)
    
    else:

        return float(np.mean(resid ** 2))


def jt_aic_from_rss(rss: float, n: int, k_params: int) -> float:
    """
    AIC under Gaussian residuals:
        AIC = 2*k + n*log(RSS/n)
    """
    mse = rss / max(1, n)
    mse = max(mse, 1e-30)
    return 2.0 * k_params + n * np.log(mse)


def jt_select_R_aic(
    X: np.ndarray,
    S: np.ndarray,
    R_grid: List[int],
    ridge_lambda: float = 1e-6,
) -> Dict[str, float]:
    """
    Select R via AIC over a candidate grid (e.g., [1,2,3,4,5,10,20]).
    """
    best = {'R': None, 'AIC': np.inf}
    table = []
    for R in R_grid:
        model = jt_fit(X, S, R=R, ridge_lambda=ridge_lambda)
        k = R + 1
        aic = jt_aic_from_rss(model['rss'], model['n'], k)
        table.append((R, aic, model['train_mse']))
        if aic < best['AIC']:
            best = {'R': R, 'AIC': float(aic)}
    best['table'] = table
    return best

def SingleNodes(S):

    M = (np.asarray(S) != 0)
    in_deg, out_deg = M.sum(0), M.sum(1)

    return np.where((in_deg==0) & (out_deg==0))[0]

def desinglarize(S):

    S = np.array(S)

    S_SingNodes = SingleNodes(S)

    S_des = S.copy()

    S_des[S_SingNodes, S_SingNodes] = 1

    return S_des

# ----------------------------- nosco (with JT-GAR) -----------------------------

def nosco_topological(
    X: np.ndarray,
    S: np.ndarray,
    R: int = 2,
    ridge_lambda: float = 1e-6,
    window_length: Optional[int] = None,
    window_overlap: float = 0.5,
    desingle: bool = False,
    reduced_error_node: bool = False,
) -> np.ndarray:
    """
    Topological nosco under JT-GAR: remove node j (from X and S) and re-fit.
    nosco_j = avg_w (E_-j - E_full) / E_full
    """
    N, T = X.shape

    # windowing
    if window_length is None:
        windows = [(0, T)]
    else:
        w = int(window_length)
        step = max(1, int(round(w*(1 - window_overlap))))
        windows = []
        t0 = 0
        while t0 + w <= T:
            windows.append((t0, t0 + w))
            t0 += step
        if not windows:
            raise ValueError("No valid windows for the given settings.")

    # full-model errors per window
    E_full = []

    if reduced_error_node:

        for (s, e) in windows:
            m_full = jt_fit(X[:, s:e], S, R=R, ridge_lambda=ridge_lambda)
            E_full_w = jt_compute_mse(X[:, s:e], m_full, return_residuals = True)
            # print(np.array(E_full_w).shape)
            E_full.append([max(1e-18, e) for e in E_full_w])
        E_full = np.array(E_full)

    else:

        for (s, e) in windows:
            m_full = jt_fit(X[:, s:e], S, R=R, ridge_lambda=ridge_lambda)
            E_full.append(jt_compute_mse(X[:, s:e], m_full))
        E_full = np.array([max(1e-18, e) for e in E_full])

    # per-node omission
    nosco = np.zeros(N, dtype=float)
    mask_all = np.ones(N, dtype=bool)
    for j in range(N):
        ratios = []
        mask = mask_all.copy(); mask[j] = False

        if desingle:
            
            S_red = desinglarize(S[np.ix_(mask, mask)])

        else:

            S_red = S[np.ix_(mask, mask)]

        for wi, (s, e) in enumerate(windows):
            X_red = X[mask, s:e]
            m_red = jt_fit(X_red, S_red, R=R, ridge_lambda=ridge_lambda)
            E_red = jt_compute_mse(X_red, m_red)

            if reduced_error_node:

                ratios.append(np.log10((E_red) / (np.mean(E_full[wi, mask]))))

            else:

                ratios.append(np.log10(E_red / E_full[wi]))

            # ratios.append((E_red - E_full[wi]) / E_full[wi])

        nosco[j] = float(np.mean(ratios))

    return nosco, np.mean(np.array(E_full), axis = 1)


def nosco_counterfactual(
    X: np.ndarray,
    S: np.ndarray,
    R: int = 2,
    ridge_lambda: float = 1e-6,
    window_length: Optional[int] = None,
    window_overlap: float = 0.5,
    counterfactual_component = 'zero', # in {'zero', 'normal_noise', 'permutation'}
) -> np.ndarray:
    """
    Counterfactual nosco under JT-GAR: keep S, clamp node j's signal to a constant, re-fit.
    C-nosco_j = avg_w (E_cf - E_full) / E_full
    """
    N, T = X.shape

    # windowing
    if window_length is None:
        windows = [(0, T)]
    else:
        w = int(window_length)
        step = max(1, int(round(w*(1 - window_overlap))))
        windows = []
        t0 = 0
        while t0 + w <= T:
            windows.append((t0, t0 + w))
            t0 += step
        if not windows:
            raise ValueError("No valid windows for the given settings.")

    # full errors
    E_full = []
    for (s, e) in windows:
        m_full = jt_fit(X[:, s:e], S, R=R, ridge_lambda=ridge_lambda)
        E_full.append(jt_compute_mse(X[:, s:e], m_full))
    E_full = np.array([max(1e-18, e) for e in E_full])

    # node-wise counterfactual suppression
    nosco_cf = np.zeros(N, dtype=float)
    for j in range(N):
        ratios = []
        for wi, (s, e) in enumerate(windows):
            Xw = X[:, s:e].copy()

            if counterfactual_component == 'permutation':

                Xw[j, :] = np.random.permutation(X[j, s:e])

            elif counterfactual_component == 'normal_noise':

                loc = np.mean(Xw[j, :])
                scale = np.var(Xw[j, :])

                Xw[j, :] = np.random.normal(loc, scale, size = (X.shape[1],))

            else:

                Xw[j, :] = np.zeros(shape = (X.shape[1],))     
                       
            m_cf = jt_fit(Xw, S, R=R, ridge_lambda=ridge_lambda)
            # E_cf = jt_compute_mse(Xw, m_cf)
            E_cf = jt_compute_mse(X[:, s:e], m_cf)
            # ratios.append((E_cf - E_full[wi]) / E_full[wi])
            ratios.append(np.log10(E_cf / E_full[wi]))
        nosco_cf[j] = float(np.mean(ratios))
    return nosco_cf, E_full

def ADer(X, beta = 1.4826):

    tmpp = []

    X = np.array(X)

    if X.ndim == 1:

        X = np.array([X])

    for X_ in X:

        adz = np.abs(X_ - np.median(X_))
        temp = np.median(adz)

        tmpp.append(adz / (beta * temp))

    return np.array(tmpp)