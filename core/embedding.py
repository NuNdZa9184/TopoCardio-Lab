"""
TopoCardio Lab — Time-delay embedding utilities.

Provides Takens embedding construction and automatic parameter selection
via Average Mutual Information (AMI) and False Nearest Neighbors (FNN).
"""

from typing import Tuple

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# 1. Takens embedding
# ---------------------------------------------------------------------------

def takens_embedding(signal: np.ndarray, delay: int, dimension: int) -> np.ndarray:
    """Construct a time-delay (Takens) embedding of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional time series of length N.
    delay : int
        Time delay (τ) between successive embedding coordinates.
    dimension : int
        Embedding dimension (m).

    Returns
    -------
    np.ndarray
        Point cloud of shape ``(N - (dimension - 1) * delay, dimension)``.

    Raises
    ------
    ValueError
        If the signal is too short for the requested parameters.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)
    n_points = N - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Signal length {N} is too short for delay={delay} and "
            f"dimension={dimension} (need at least {(dimension - 1) * delay + 1} samples)."
        )

    # Build the embedding matrix via stride tricks for efficiency
    indices = np.arange(dimension) * delay          # shape (dimension,)
    row_offsets = np.arange(n_points)[:, None]      # shape (n_points, 1)
    return signal[row_offsets + indices]             # shape (n_points, dimension)


# ---------------------------------------------------------------------------
# 2. Average Mutual Information (AMI)
# ---------------------------------------------------------------------------

def compute_ami(
    signal: np.ndarray,
    max_lag: int = 50,
    n_bins: int = 64,
) -> Tuple[np.ndarray, int]:
    """Compute the Average Mutual Information for a range of lags.

    Uses a histogram-based estimator of mutual information between
    ``signal[:-lag]`` and ``signal[lag:]`` for each lag in ``1 .. max_lag``.

    The optimal delay τ is chosen as the **first local minimum** of the AMI
    curve.  If no local minimum exists, the lag with the global minimum is
    returned instead.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional time series.
    max_lag : int, optional
        Maximum lag to evaluate (default 50).
    n_bins : int, optional
        Number of histogram bins along each axis (default 64).

    Returns
    -------
    ami_values : np.ndarray
        Array of shape ``(max_lag,)`` with AMI values for lags 1 … max_lag.
    optimal_tau : int
        The lag (≥ 1) at the first local minimum of the AMI curve.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)

    if max_lag >= N:
        raise ValueError(
            f"max_lag={max_lag} must be smaller than signal length {N}."
        )

    ami_values = np.empty(max_lag, dtype=np.float64)

    # Pre-compute the common bin edges from the full signal range
    s_min, s_max = signal.min(), signal.max()
    edges = np.linspace(s_min, s_max, n_bins + 1)
    # Slight widening so that max value falls inside the last bin
    edges[-1] += 1e-10 * (s_max - s_min + 1e-15)

    for lag_idx, lag in enumerate(range(1, max_lag + 1)):
        x = signal[: N - lag]
        y = signal[lag:]

        # Joint and marginal histograms
        joint, _, _ = np.histogram2d(x, y, bins=[edges, edges])
        joint = joint / joint.sum()  # normalise to joint probability

        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # Mutual information  I(X;Y) = Σ p(x,y) log[ p(x,y) / (p(x) p(y)) ]
        # Only sum where the joint probability is non-zero
        nz = joint > 0
        mi = np.sum(
            joint[nz] * np.log(joint[nz] / (px[:, None] * py[None, :])[nz])
        )
        ami_values[lag_idx] = mi

    # --- Find the first local minimum ---
    optimal_tau = _first_local_minimum(ami_values, start_index=0) + 1  # +1 → lag (1-based)

    return ami_values, int(optimal_tau)


def _first_local_minimum(values: np.ndarray, start_index: int = 0) -> int:
    """Return the index of the first local minimum in *values*.

    A point ``values[i]`` is a local minimum if
    ``values[i-1] >= values[i] <= values[i+1]``.

    If no local minimum is found, returns the index of the global minimum.
    """
    for i in range(max(start_index, 1), len(values) - 1):
        if values[i - 1] >= values[i] and values[i] <= values[i + 1]:
            return i
    # Fallback: global minimum
    return int(np.argmin(values[start_index:])) + start_index


# ---------------------------------------------------------------------------
# 3. False Nearest Neighbors (FNN)
# ---------------------------------------------------------------------------

def compute_fnn(
    signal: np.ndarray,
    delay: int,
    max_dim: int = 6,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> Tuple[np.ndarray, int]:
    """Estimate the fraction of False Nearest Neighbors for dimensions 1 … max_dim.

    For each candidate embedding dimension *d* the algorithm:

    1. Embeds the signal in *d* dimensions.
    2. Finds the nearest neighbour of every point using a KDTree.
    3. Re-embeds in *d + 1* dimensions and checks whether the neighbour
       relationship is an artefact of too-low dimensionality, using two
       criteria (Kennel, Brown & Abarbanel, 1992):

       * **rtol criterion** – the distance increase ratio exceeds *rtol*.
       * **atol criterion** – the new-dimension separation exceeds
         ``atol × R_A`` where ``R_A`` is the standard deviation (attractor
         size) of the signal.

       A neighbour that fails *either* criterion is counted as false.

    The optimal dimension is the smallest *d* for which the FNN fraction
    drops below 1 %.  If none qualifies, ``max_dim`` is returned.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional time series.
    delay : int
        Time delay τ for the embedding.
    max_dim : int, optional
        Maximum embedding dimension to test (default 6).
    rtol : float, optional
        Distance-ratio tolerance (default 15.0).
    atol : float, optional
        Absolute tolerance relative to attractor size (default 2.0).

    Returns
    -------
    fnn_fractions : np.ndarray
        Array of shape ``(max_dim,)`` — fraction of FNN for dimensions
        1 … max_dim.
    optimal_dim : int
        Smallest dimension (≥ 1) where the FNN fraction < 0.01.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    R_A = np.std(signal)  # attractor size estimate
    atol_abs = atol * R_A

    fnn_fractions = np.empty(max_dim, dtype=np.float64)

    for dim in range(1, max_dim + 1):
        # Embed in dimension d and d+1
        emb_d = takens_embedding(signal, delay, dim)
        emb_d1 = takens_embedding(signal, delay, dim + 1)

        # Both must share the same set of time indices.
        # emb_d has more rows; trim to match emb_d1.
        n_pts = emb_d1.shape[0]
        emb_d = emb_d[:n_pts]

        # Build KDTree in d-dimensional space and query nearest neighbour
        # k=2 because the first hit is the point itself when using the same
        # array — but here indices are unique rows, so k=1 would suffice
        # only if no duplicate rows exist. We use k=2 to be safe, taking
        # the second neighbour if the first is the point itself.
        tree = KDTree(emb_d)
        dists, idxs = tree.query(emb_d, k=2)
        # Nearest neighbour is column 1 (column 0 is the point itself with dist 0)
        nn_dists = dists[:, 1]
        nn_idxs = idxs[:, 1]

        # --- Check FNN criteria ---
        # Additional coordinate in the (d+1)-th dimension
        extra_coord_self = emb_d1[:, -1]
        extra_coord_nn = emb_d1[nn_idxs, -1]
        delta = np.abs(extra_coord_self - extra_coord_nn)

        # Avoid division by zero for identical neighbours
        safe_dists = np.where(nn_dists > 0, nn_dists, np.inf)

        criterion_1 = (delta / safe_dists) > rtol
        criterion_2 = delta > atol_abs
        is_false = criterion_1 | criterion_2

        fnn_fractions[dim - 1] = np.mean(is_false)

    # --- Optimal dimension: first dim where FNN < 1 % ---
    optimal_dim = max_dim  # fallback
    for dim_idx in range(max_dim):
        if fnn_fractions[dim_idx] < 0.01:
            optimal_dim = dim_idx + 1  # 1-based dimension
            break

    return fnn_fractions, int(optimal_dim)
