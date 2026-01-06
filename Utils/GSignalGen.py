import numpy as np
import numpy.linalg as npl
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple, Dict
import re
import scipy.stats as sps

Array = np.ndarray

# ------------------------- Shift normalization -------------------------

def normalize_shift(A: Array, make_symmetric: bool = True, eig_normalized = False, prevent_single_node = False) -> Array:
    """
    Symmetric normalized adjacency:
        S = D^{-1/2} A D^{-1/2}
    where D_ii = sum_j A_ij. For undirected graphs, S is symmetric and ||S||2 <= 1.
    """
    A = np.asarray(A, dtype=float)
    if make_symmetric and not eig_normalized:

        A = 0.5 * (A + A.T)

    if prevent_single_node:

        A_SingleNodes = SingleNodes(A)

        A[A_SingleNodes, A_SingleNodes] = 1

    else:

        np.fill_diagonal(A, 0.0)

    if not eig_normalized:

        d1 = np.sum(A, axis = 1)
        d0 = np.sum(A, axis = 0)

        inv_sqrt_d0 = 1.0 / np.sqrt(np.maximum(d0, 1e-12))
        inv_sqrt_d1 = 1.0 / np.sqrt(np.maximum(d1, 1e-12))

        D0_inv_sqrt = np.diag(inv_sqrt_d0)
        D1_inv_sqrt = np.diag(inv_sqrt_d1)

        return D0_inv_sqrt @ A @ D1_inv_sqrt
    
    else:

        _, SingularValues, _ = np.linalg.svd(A)
        SigMax = np.max(SingularValues)

        return A / SigMax

    # U, _, Vh = np.linalg.svd(A)

    # return U @ Vh

def SingleNodes(S):

    if sp.issparse(S):

        in_deg  = np.ravel((S != 0).sum(axis=0))
        out_deg = np.ravel((S != 0).sum(axis=1))

    else:

        M = (np.asarray(S) != 0)
        in_deg, out_deg = M.sum(0), M.sum(1)

    return np.where((in_deg==0) & (out_deg==0))[0]

# ------------- Non-Linear Energy Conserving Shift Operator --------------

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def permutation_for_block_diag(W, threshold=0.0, mode='weak', return_labels=False, sparse_P=False):
    """
    Compute a permutation matrix P such that P.T @ W @ P groups nodes by components.
    If the current ordering already yields block-diagonal structure (components are contiguous),
    return the identity matrix instead of a nontrivial permutation.

    Parameters
    ----------
    W : (N,N) array-like or sparse matrix
        Adjacency/weight matrix; can be weighted or unweighted, directed or undirected.
    threshold : float, default 0.0
        Edge exists iff |W_ij| > threshold.
    mode : {'undirected','weak','strong'}, default 'weak'
        - 'undirected' : treat as undirected by symmetrizing (A | A.T); guarantees block diagonal.
        - 'weak'       : weakly connected components for directed graphs (uses connection='weak'); block diagonal.
        - 'strong'     : strongly connected components; P.T @ W @ P is block *upper-triangular* in general.
    return_labels : bool, default False
        If True, also return the component label per node and the order.
    sparse_P : bool, default True
        If True, return P as a sparse CSC permutation matrix; else return a dense ndarray.

    Returns
    -------
    P : (N,N) permutation matrix (sparse or dense)
        Identity if components are already contiguous (block-diagonal in current order),
        else a permutation that groups nodes by component.
    (optional) labels : (N,) int
        Component id per node (0..K-1).
    (optional) order : (N,) int
        The permutation vector such that W[np.ix_(order, order)] is block-ordered.
    """
    # Convert to sparse for graph ops
    Wsp = sp.csr_matrix(W) if not sp.issparse(W) else W.tocsr()
    n = Wsp.shape[0]

    # Binarize by threshold on absolute value
    B = Wsp.copy()
    if B.nnz:
        B.data = (np.abs(B.data) > threshold).astype(np.int8)
        B.eliminate_zeros()

    # Pick connectivity interpretation
    if mode == 'undirected':
        # Symmetrize: edge exists if either direction exists
        B = sp.csr_matrix(sp.maximum(B, B.T))
        _, labels = connected_components(csgraph=B, directed=False)
    elif mode == 'weak':
        # Weak components (direction ignored internally)
        _, labels = connected_components(csgraph=B, directed=True, connection='weak')
    elif mode == 'strong':
        # Strongly connected components (may yield block upper-triangular, not diagonal)
        _, labels = connected_components(csgraph=B, directed=True, connection='strong')
    else:
        raise ValueError("mode must be one of {'undirected','weak','strong'}")

    # --- Check if labels already form contiguous blocks in the current order ---
    seen = set()
    contiguous = True
    last_label = None
    for i in range(n):
        lb = labels[i]
        if lb != last_label:
            # starting a new run
            if lb in seen:
                # this label appeared before and reappears -> not contiguous
                contiguous = False
                break
            seen.add(lb)
            last_label = lb

    if contiguous:
        # Current ordering already block-diagonal by components -> identity permutation
        if sparse_P:
            P = sp.identity(n, format='csc')
        else:
            P = np.eye(n)
        if return_labels:
            order = np.arange(n)
            return P, labels, order
        return P

    # --- Otherwise, build permutation that groups nodes by component id ---
    order = np.argsort(labels)
    if sparse_P:
        # P = I[:, order] as sparse
        P = sp.identity(n, format='csc')[:, order]
    else:
        P = np.eye(n)[:, order]

    if return_labels:
        return P, labels, order
    return P


def split_block_diagonal(A, threshold: float = 0.0):
    """
    Decompose a block-diagonal matrix A (NxN) into a list of NxN matrices, each
    containing exactly one diagonal block in its original positions.

    Parameters
    ----------
    A : (N,N) array-like or sparse matrix
    threshold : float, default 0.0
        Entry is considered nonzero iff |A_ij| > threshold.

    Returns
    -------
    blocks : list of (N,N) matrices (same type as A)
    comp_indices : list of 1D ndarray
    """
    Asp = sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    n = Asp.shape[0]

    # Binarize by threshold on absolute value
    B = Asp.copy()
    if B.nnz:
        B.data = (np.abs(B.data) > threshold).astype(np.int8)
        B.eliminate_zeros()

    # Symmetrize the support correctly (use the *method*, not sp.maximum function)
    Bsym = B.maximum(B.T).tocsr()
    # Make sure it's binary (optional but nice)
    if Bsym.nnz:
        Bsym.data[:] = 1

    # Connected components → each diagonal block’s index set
    n_comp, labels = connected_components(csgraph=Bsym, directed=False)

    comp_ids = sorted(range(n_comp), key=lambda c: np.flatnonzero(labels == c)[0])
    blocks, comp_indices = [], []

    if sp.issparse(A):
        for c in comp_ids:
            idx = np.flatnonzero(labels == c)
            sub = Asp[idx, :][:, idx].tocoo()
            rows = idx[sub.row]
            cols = idx[sub.col]
            data = sub.data
            A_block = sp.coo_matrix((data, (rows, cols)), shape=Asp.shape).tocsr()
            blocks.append(A_block.tocsc())
            comp_indices.append(idx)
    else:
        A = np.asarray(A)
        for c in comp_ids:
            idx = np.flatnonzero(labels == c)
            Ab = np.zeros_like(A)
            Ab[np.ix_(idx, idx)] = A[np.ix_(idx, idx)]
            blocks.append(Ab)
            comp_indices.append(idx)

    return blocks, comp_indices      

def SingleNodes(S):
    if sp.issparse(S):
        in_deg  = np.ravel((S != 0).sum(axis=0))
        out_deg = np.ravel((S != 0).sum(axis=1))
    else:
        M = (np.asarray(S) != 0)
        in_deg, out_deg = M.sum(0), M.sum(1)
    return np.where((in_deg==0) & (out_deg==0))[0]

def ShiftOperator(S, n, x, preserve_energy = True):

    # print([S.shape, x.shape])

    single_nodes = SingleNodes(S)

    S_m = S.copy()
    S_m[single_nodes, single_nodes] = 1

    P_BD, labels, order = permutation_for_block_diag(S_m, return_labels = True)

    S_BD = P_BD.T @ S_m @ P_BD
    x_BD = P_BD.T @ x.copy()

    NumComps = len(np.unique(labels))

    S_s, Indices = split_block_diagonal(S_BD)

    X_s = []

    for Indice in Indices:

        x_tmp = np.zeros_like(x_BD)
        x_tmp[Indice] = x_BD[Indice]

        X_s.append(x_tmp)

    X_Energy = [np.sqrt(np.sum(X ** 2)) for X in X_s]

    S_Powers = [np.linalg.matrix_power(S, n) for S in S_s]

    X_shifted = []

    for Component in range(len(Indices)):

        X_shifted_tmp = S_Powers[Component] @ X_s[Component]

        if preserve_energy:

            X_st_Energy = np.sqrt(np.sum(X_shifted_tmp ** 2))

            X_shifted_tmp = X_shifted_tmp / (X_st_Energy + 1e-12) * X_Energy[Component]

        X_shifted.append(X_shifted_tmp)

    return P_BD @ np.sum(np.array(X_shifted), axis = 0)

# ------------------------- Parsing driver specs -------------------------

def parse_driver_spec(spec: str) -> Dict:
    """
    Supported:
      - spike-{p100}-{kernel_len?}       e.g., 'spike-3' or 'spike-3-7'
      - sin-{freq}-{amp}                 e.g., 'sin-6-1.0'   (freq in Hz)
      - square-{period}-{duty}           e.g., 'square-80-0.3'  (period in samples)
      - AR-{order}-{coeffs?}             e.g., 'AR-3-0.6,0.2,0.1' or 'AR-3'
    """
    s = spec.strip().lower()
    if s.startswith("spike-"):
        parts = s.split("-")
        if len(parts) < 2:
            raise ValueError(f"Bad spike spec: {spec}")
        p100 = float(parts[1])
        kernel_len = int(parts[2]) if len(parts) >= 3 else None
        return {"type": "spike", "prob_per_100": p100, "kernel_len": kernel_len}
    if s.startswith("sin-"):
        m = re.match(r"sin\-([0-9]+(\.[0-9]+)?)\-([0-9]+(\.[0-9]+)?)", s)
        if not m: raise ValueError(f"Bad sinusoid spec: {spec}")
        return {"type": "sin", "freq": float(m.group(1)), "amp": float(m.group(3))}
    if s.startswith("square-"):
        m = re.match(r"square\-([0-9]+)\-([0-1](\.[0-9]+)?)", s)
        if not m: raise ValueError(f"Bad square spec: {spec}")
        return {"type": "square", "period": int(m.group(1)), "duty": float(m.group(2))}
    if s.startswith("ar-"):
        m = re.match(r"ar\-([0-9]+)(\-(.*))?", s)
        if not m: raise ValueError(f"Bad AR spec: {spec}")
        order = int(m.group(1))
        coeffs = None
        if m.group(3):
            coeffs = [float(c) for c in m.group(3).split(",")]
            if len(coeffs) != order:
                raise ValueError("AR coeffs length must match the order.")
        return {"type": "ar", "order": order, "coeffs": coeffs}
    raise ValueError(f"Unknown driver spec: {spec}")

# ------------------------- Helpers -------------------------

def exp_smoothing_kernel(length: int, beta: float = 0.6) -> Array:
    """Causal exponentially-decaying kernel (sum=1)."""
    length = max(1, int(length))
    k = beta ** np.arange(length)
    k = k / np.sum(k)
    return k

def make_stable_ar_coeffs(order: int, rng: np.random.Generator, max_abs: float = 0.8) -> Array:
    """
    Heuristic stable AR coeffs: small, decaying magnitudes. Not exact root placement,
    but adequate for synthetic drivers.
    """
    a = rng.uniform(-max_abs, max_abs, size=order) * (0.9 ** np.arange(order))
    if np.sum(np.abs(a)) > 0.95:
        a *= 0.9 / (np.sum(np.abs(a)) + 1e-12)
    return a

def enforce_nonexpansive(b: List[float], slack: float = 1e-2) -> List[float]:
    """
    Project diffusion coefficients b_r to satisfy sum |b_r| <= 1 - slack.
    Ensures the graph-time operator is non-expansive when ||S||2 <= 1.
    """
    b = np.array(b, dtype=float)
    s = np.sum(np.abs(b))
    if s > 1.0 - slack and s > 0:
        b = b * ((1.0 - slack) / s)
    return b.tolist()

def eigen_normalize(M):

    e, V = np.linalg.eig(M)
    e1 = e / np.sqrt(np.sum(e ** 2))

    return np.real(np.real(V) @ np.diag(e1) @ np.real(V.T))

# ------------------------- Main class -------------------------

@dataclass
class GraphSignalSimulator:
    """
    Energy-safe graph signal simulator using symmetric normalized shift S = D^{-1/2} A D^{-1/2}.
    Dynamics (non-expansive by design):
        x_t = sum_{r=1..R} b_r * S^r * x_{t-1} + U[:, t]
    with ||S||2 <= 1 and sum |b_r| <= 1 - eps  ==> energy does not blow up.

    - Accepts adjacency A (undirected recommended).
    - Allows multiple driver nodes and driver specs.
    - Global SNR control via post-additive AWGN (optional).
    """
    A: Array                                  # Adjacency (prefer undirected, unweighted/weighted)
    T: int = 10000                            # number of time samples
    fs: float = 100.0                         # sampling frequency (Hz) for sinusoidal drivers
    R: int = 1                                # graph diffusion order
    b_graph: Optional[List[float]] = None     # diffusion coefficients for S^r
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1234))
    make_symmetric: bool = True               # symmetrize A before normalization
    epsilon_nonexp: float = 1e-2              # slack for non-expansiveness
    energy_preserve: bool = True
    eig_normalize: bool = False
    prevent_single_node: bool = False

    def __post_init__(self):
        self.S: Array = normalize_shift(self.A, make_symmetric=self.make_symmetric, prevent_single_node = self.prevent_single_node,
                                        eig_normalized = self.eig_normalize)
        # Default b_r (decaying), then enforce non-expansive sum
        if self.b_graph is None:
            self.b_graph = [0.6] + [0.2] * (max(0, self.R - 1))
        self.b_graph = enforce_nonexpansive(self.b_graph, slack=self.epsilon_nonexp)
        self.EnPr = self.energy_preserve

    # ---------- Driver builders ----------

    def driver_spike(self, p100: float = 3.0, kernel_len: Optional[int] = None) -> Array:
        """
        Random spikes: approx p100 spikes per 100-sample block.
        If kernel_len is provided, smooth with an exponential kernel of that length.
        """
        x = np.zeros(self.T, dtype=float)
        num_blocks = self.T // 100
        for b in range(num_blocks):
            k = int(max(1, round(p100)))
            idx = self.rng.choice(np.arange(b*100, (b+1)*100), size=k, replace=False)
            x[idx] = 1.0
        tail = self.T - num_blocks*100
        if tail >= 30 and p100 >= 1:
            kt = max(1, int(round(p100/2)))
            idx = self.rng.choice(np.arange(num_blocks*100, self.T), size=kt, replace=False)
            x[idx] = 1.0
        if kernel_len is not None and kernel_len > 0:
            k = exp_smoothing_kernel(kernel_len, beta=0.6)
            y = np.convolve(x, k, mode="full")[:self.T]
            return y
        return x

    def driver_sinusoid(self, freq: float = 6.0, amp: float = 1.0, phase: float = 0.0) -> Array:
        t = np.arange(self.T) / self.fs
        return amp * np.sin(2*np.pi*freq*t + phase)

    def driver_square(self, period: int = 80, duty: float = 0.5, amp: float = 1.0, phase_shift: int = 0) -> Array:
        period = max(1, int(period))
        duty = float(np.clip(duty, 0.0, 1.0))
        x = np.zeros(self.T, dtype=float)
        idx = (np.arange(self.T) + int(phase_shift)) % period
        x[idx < int(round(duty * period))] = amp
        return x

    def driver_ar(self, order: int = 3, coeffs: Optional[List[float]] = None, noise_std: float = 1.0) -> Array:
        if coeffs is None:
            coeffs = make_stable_ar_coeffs(order, self.rng).tolist()
        a = np.array(coeffs, dtype=float)
        y = np.zeros(self.T, dtype=float)
        for t in range(order, self.T):
            y[t] = float(np.dot(a, y[t-order:t][::-1])) + self.rng.normal(0.0, noise_std)
        std = np.std(y)
        if std > 1e-9: y /= std
        return y

    def make_driver_from_spec(self, spec: str) -> Array:
        d = parse_driver_spec(spec)
        if d["type"] == "spike":
            return self.driver_spike(p100=float(d["prob_per_100"]), kernel_len=d["kernel_len"])
        if d["type"] == "sin":
            return self.driver_sinusoid(freq=float(d["freq"]), amp=float(d["amp"]))
        if d["type"] == "square":
            return self.driver_square(period=int(d["period"]), duty=float(d["duty"]), amp=1.0)
        if d["type"] == "ar":
            return self.driver_ar(order=int(d["order"]), coeffs=d["coeffs"])
        raise ValueError("Unsupported driver type.")

    # ---------- Simulation ----------

    def simulate(
        self,
        driver_nodes: Union[int, List[int]],
        driver_specs: Union[str, List[str]],
        snr_db: Optional[float] = None,              # sensing noise (as before)
        copy_behavior: bool = False,
        injection_gain: float = 1.0,
        transmission_noise: bool = False,            # new flag
        transmission_snr_db: float = 10.0,           # default SNR for transmission noise
    ) -> Tuple[Array, Array]:

        N = self.S.shape[0]
        nodes = [driver_nodes] if isinstance(driver_nodes, int) else list(driver_nodes)

        if isinstance(driver_specs, str):
            specs = [driver_specs] * len(nodes)
        else:
            specs = list(driver_specs)
            if len(specs) == 1 and len(nodes) > 1 and not copy_behavior:
                specs = specs * len(nodes)
            elif len(specs) != len(nodes):
                raise ValueError("Length of driver_specs must equal driver_nodes (or provide a single spec).")

        U = np.zeros((N, self.T), dtype=float)
        base_waveform = None
        for i, node in enumerate(nodes):
            spec = specs[i]
            if len(nodes) > 1 and isinstance(driver_specs, str) and copy_behavior:
                if base_waveform is None:
                    base_waveform = self.make_driver_from_spec(spec)
                wave = base_waveform
            else:
                wave = self.make_driver_from_spec(spec)
            U[node, :] += injection_gain * wave

        # Precompute S^r
        Spows: List[Array] = []
        cur = self.S.copy()
        
        for r in range(1, self.R + 1):
            if r == 1:
                cur = self.S
            else:
                cur = cur @ self.S

            Spows.append(cur)

        # Dynamics with optional transmission noise
        X_clean = np.zeros((N, self.T), dtype=float)
        for t in range(1, self.T):
            diff = np.zeros(N, dtype=float)
            for r, Sr in enumerate(Spows, start=1):
                br = self.b_graph[r-1] if (r-1) < len(self.b_graph) else 0.0
                if br != 0.0:

                    # if self.EnPr:

                    #     diff += br * ShiftOperator(Sr, 1, X_clean[:, t - 1])

                    # else:

                        diff += br * (Sr @ X_clean[:, t-1])

            signal_part = diff + U[:, t]

            if transmission_noise:
                sig_pow = float(np.mean(signal_part**2))
                if sig_pow > 1e-30:
                    noise_pow = sig_pow / (10.0 ** (transmission_snr_db / 10.0))
                    noise_std = np.sqrt(noise_pow)
                    signal_part = signal_part + self.rng.normal(0.0, noise_std, size=signal_part.shape)

            X_clean[:, t] = signal_part # / np.max([np.sqrt(np.sum(signal_part ** 2)), 1e-12])

        # Optional sensing noise (as before)
        if snr_db is None:
            X = X_clean.copy()
        else:
            sig_pow = float(np.mean(X_clean**2))
            if sig_pow <= 1e-30:
                X = X_clean.copy()
            else:
                noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
                noise_std = np.sqrt(noise_pow)
                X = X_clean + self.rng.normal(0.0, noise_std, size=X_clean.shape)

        return X, U


def generate_random_graph_matrix(
    N: int = 10,
    is_weighted: bool = False,
    is_directed: bool = False,
    output_matrix: str = "Adj",
    weights_dist: str = "Uniform",
    weights_range: tuple = (0.0, 1.0),
    p_edge: float = 0.3,
    seed: int = None
) -> np.ndarray:
    """
    Generate a random adjacency or Laplacian matrix.

    Parameters
    ----------
    N : int, default=10
        Number of nodes.
    is_weighted : bool, default=False
        If True, assign random weights to edges.
    is_directed : bool, default=False
        If False, adjacency is symmetric (undirected).
    output_matrix : {'Adj', 'Lap'}, default='Adj'
        Whether to return adjacency or Laplacian.
    weights_dist : str, default='Uniform'
        Distribution for weights if is_weighted=True.
        - 'Uniform' : uniform in weights_range.
        - 'Gaussian-m-v' : Gaussian with mean=m, var=v.
    weights_range : tuple, default=(0,1)
        Min/max for Uniform distribution.
    p_edge : float, default=0.3
        Probability of an edge (Bernoulli).
    seed : int or None
        RNG seed.

    Returns
    -------
    M : (N,N) ndarray
        Adjacency or Laplacian matrix.
    """
    rng = np.random.default_rng(seed)

    # Step 1: adjacency mask (0/1)
    A = (rng.random((N, N)) < p_edge).astype(float)

    if not is_directed:
        A = np.triu(A, 1)
        A = A + A.T

    np.fill_diagonal(A, 0.0)

    # Step 2: add weights if needed
    if is_weighted:
        if weights_dist.lower() == "uniform":
            w = rng.uniform(weights_range[0], weights_range[1], size=A.shape)
        elif weights_dist.lower().startswith("gaussian"):
            parts = weights_dist.split("-")
            mean, var = float(parts[1]), float(parts[2])
            w = rng.normal(mean, np.sqrt(var), size=A.shape)
            w = np.clip(w, 0, None)  # optional: keep nonnegative
        else:
            raise ValueError("Unknown weights_dist")
        A = A * w

    # Step 3: Laplacian if needed
    if output_matrix.lower() == "lap":
        d = np.sum(A, axis=1)
        L = np.diag(d) - A
        return L
    else:
        return A

def star_adjacency(N: int, hub: int = 0) -> np.ndarray:
    """
    Build adjacency matrix of an undirected star graph with N nodes.
    hub: index of the central node (0 <= hub < N).
    Returns an (N x N) int matrix.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if not (0 <= hub < N):
        raise ValueError("hub must be in the range [0, N-1].")

    A = np.zeros((N, N), dtype=int)
    if N == 1:
        return A  # single node, no edges

    # connect hub to all others (undirected)
    A[hub, :] = 0
    A[:, hub] = 0
    for i in range(N):
        if i == hub:
            continue
        A[hub, i] = 1
        A[i, hub] = 1
    return A

import numpy as np

# Helper برای وزن‌دهی و جهت‌دار کردن
def _finalize_adj(A, directed=False, weighted=False, weight_range=(1.0, 1.0), seed=None):
    rng = np.random.default_rng(seed)
    N = A.shape[0]

    if directed:
        # هر یال غیرقطری رو فقط در یک جهت نگه می‌داریم
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] == 1 and A[j, i] == 1:
                    if rng.random() < 0.5:
                        A[j, i] = 0
                    else:
                        A[i, j] = 0

    if weighted:
        low, high = weight_range
        weights = rng.uniform(low, high, size=A.shape)
        A = A * weights

    return A.astype(float if weighted else int)


def star_adj_mat(N, hub=0, directed=False, weighted=False, weight_range=(1,1), seed=None):
    A = np.zeros((N, N))
    for i in range(N):
        if i != hub:
            A[hub, i] = 1
            A[i, hub] = 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def path_adj_mat(N, directed=False, weighted=False, weight_range=(1,1), seed=None):
    A = np.zeros((N, N))
    for i in range(N-1):
        A[i, i+1] = A[i+1, i] = 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def cycle_adj_mat(N, directed=False, weighted=False, weight_range=(1,1), seed=None):
    if N < 3:
        raise ValueError("Cycle graph needs N >= 3")
    A = path_adj_mat(N)
    A[0, N-1] = A[N-1, 0] = 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def wheel_adj_mat(N, directed=False, weighted=False, weight_range=(1,1), seed=None):
    if N < 4:
        raise ValueError("Wheel graph needs N >= 4")
    A = cycle_adj_mat(N-1)  # چرخه روی N-1 نود اول
    hub = N-1
    A = np.pad(A, ((0,1),(0,1)))  # اضافه کردن نود هاب
    for i in range(N-1):
        A[hub, i] = A[i, hub] = 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def bipartite_adj_mat(N, n1=None, complete=False, p=0.5,
                      directed=False, weighted=False, weight_range=(1,1), seed=None):
    rng = np.random.default_rng(seed)
    if n1 is None:
        n1 = rng.integers(1, N)  # بین 1 و N-1
    n2 = N - n1
    left = range(n1)
    right = range(n1, N)
    A = np.zeros((N, N))
    for i in left:
        for j in right:
            if complete or rng.random() < p:
                A[i, j] = A[j, i] = 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def regular_adj_mat(N, k=2, directed=False, weighted=False, weight_range=(1,1), seed=None):
    if k >= N:
        raise ValueError("k must be less than N")
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    degrees = [0]*N
    while min(degrees) < k:
        i, j = rng.choice(N, 2, replace=False)
        if A[i, j] == 0 and degrees[i] < k and degrees[j] < k:
            A[i, j] = A[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def tree_adj_mat(N, directed=False, weighted=False, weight_range=(1,1), seed=None):
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    for i in range(1, N):
        j = rng.integers(0, i)  # وصل کردن به یکی از قبلیا
        A[i, j] = A[j, i] = 1
    return _finalize_adj(A, directed, weighted, weight_range, seed)


def forest_adj_mat(N, n_components=2, directed=False, weighted=False, weight_range=(1,1), seed=None):
    rng = np.random.default_rng(seed)
    if n_components < 1 or n_components > N:
        raise ValueError("n_components must be between 1 and N")
    sizes = [N // n_components] * n_components
    for i in range(N % n_components):
        sizes[i] += 1
    A = np.zeros((N, N))
    start = 0
    for size in sizes:
        subA = tree_adj_mat(size, seed=int(rng.integers(1e9)))
        A[start:start+size, start:start+size] = subA
        start += size
    return _finalize_adj(A, directed, weighted, weight_range, seed)

import numpy as np

def regular_adj_mat_c(N, k, directed=False, weighted=False, weight_range=(1.0, 1.0), seed=None):
    if not (0 <= k < N):
        raise ValueError("0 <= k < N must hold")
    if not directed and (N * k) % 2 != 0:
        raise ValueError("For undirected simple k-regular graphs, N*k must be even.")
    if not directed and (k % 2 == 1) and (N % 2 == 1):
        raise ValueError("For odd k, N must be even (undirected case).")

    A = np.zeros((N, N), dtype=int)
    idx = np.arange(N)

    if not directed:
        # even k: connect ±1..±(k/2)
        if k % 2 == 0:
            for t in range(1, k // 2 + 1):
                jdx = (idx + t) % N
                A[idx, jdx] = 1
                A[jdx, idx] = 1
        else:
            # odd k: need N even; connect ±1..±((k-1)/2) and the opposite node N/2
            for t in range(1, (k - 1) // 2 + 1):
                jdx = (idx + t) % N
                A[idx, jdx] = 1
                A[jdx, idx] = 1
            jdx = (idx + N // 2) % N
            A[idx, jdx] = 1
            A[jdx, idx] = 1
    else:
        # directed k-in/k-out-regular: edges i -> i+t (t=1..k)
        for t in range(1, k + 1):
            jdx = (idx + t) % N
            A[idx, jdx] = 1
        # توجه: برای k < N/2 دو-چرخه نداریم؛ برای k بزرگ ممکن است i->j و j->i هر دو باشند.

    if weighted:
        rng = np.random.default_rng(seed)
        low, high = weight_range
        W = rng.uniform(low, high, size=A.shape)
        A = A * W
        return A.astype(float)
    return A
