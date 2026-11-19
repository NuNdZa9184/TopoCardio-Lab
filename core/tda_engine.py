"""
TopoCardio Lab — TDA Engine
============================
Core topological data analysis computations: persistence diagrams,
registered scalar metrics, persistent entropy, and AAFT surrogate testing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as scipy_stats
from ripser import ripser
from persim import wasserstein as persim_wasserstein, bottleneck as persim_bottleneck

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, dict] = {}


def register_metric(name: str, description: str = ""):
    """Decorator that registers a TDA scalar metric by *name*."""
    def decorator(func):
        _METRIC_REGISTRY[name] = {"func": func, "description": description}
        return func
    return decorator


def get_all_metrics() -> dict:
    """Return a shallow copy of the full metric registry."""
    return dict(_METRIC_REGISTRY)


def compute_metric(name: str, **kwargs):
    """Look up *name* in the registry and call the metric function."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. "
                       f"Available: {list(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]["func"](**kwargs)


# ---------------------------------------------------------------------------
# Core persistence computation
# ---------------------------------------------------------------------------

def compute_persistence(point_cloud: ArrayLike, maxdim: int = 1) -> dict:
    """Run Vietoris–Rips persistence via *ripser*.

    Parameters
    ----------
    point_cloud : array-like, shape (n_points, n_dims)
        The point cloud (e.g. a Takens embedding).
    maxdim : int
        Maximum homology dimension to compute.

    Returns
    -------
    dict
        Ripser result dictionary.  ``result['dgms']`` holds a list of
        persistence diagrams (one ``ndarray`` per dimension).
    """
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    return ripser(point_cloud, maxdim=maxdim)


def get_diagram(result: dict, dim: int) -> np.ndarray:
    """Extract the persistence diagram for homology dimension *dim*.

    Parameters
    ----------
    result : dict
        Ripser result dictionary (must contain key ``'dgms'``).
    dim : int
        Homology dimension (0, 1, …).

    Returns
    -------
    np.ndarray, shape (n_features, 2)
        Each row is ``[birth, death]``.
    """
    dgms = result["dgms"]
    if dim >= len(dgms):
        raise ValueError(
            f"Dimension {dim} not available; "
            f"result contains dimensions 0–{len(dgms) - 1}."
        )
    return dgms[dim]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_finite(dgm: np.ndarray) -> np.ndarray:
    """Return only rows where both birth and death are finite."""
    dgm = np.asarray(dgm, dtype=np.float64)
    if dgm.ndim != 2 or dgm.shape[0] == 0:
        return dgm
    mask = np.all(np.isfinite(dgm), axis=1)
    return dgm[mask]


def _lifetimes(dgm: np.ndarray) -> np.ndarray:
    """Compute finite lifetimes (death − birth) from a diagram."""
    dgm = _filter_finite(dgm)
    if dgm.shape[0] == 0:
        return np.array([], dtype=np.float64)
    return dgm[:, 1] - dgm[:, 0]


def persistent_entropy(dgm: np.ndarray) -> float:
    """Shannon entropy of a single persistence diagram.

    .. math::

        \\ell_i = d_i - b_i, \\quad
        p_i = \\frac{\\ell_i}{\\sum_j \\ell_j}, \\quad
        H = -\\sum_i p_i \\ln p_i

    Returns 0.0 for empty diagrams.
    """
    lt = _lifetimes(dgm)
    lt = lt[lt > 0]
    if lt.size == 0:
        return 0.0
    total = lt.sum()
    p = lt / total
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Registered metrics
# ---------------------------------------------------------------------------

@register_metric("max_h1_persistence",
                  description="Maximum lifetime in H1")
def max_h1_persistence(dgms: list[np.ndarray], **_kw) -> float:
    lt = _lifetimes(dgms[1])
    return float(lt.max()) if lt.size > 0 else 0.0


@register_metric("mean_h1_persistence",
                  description="Mean lifetime in H1")
def mean_h1_persistence(dgms: list[np.ndarray], **_kw) -> float:
    lt = _lifetimes(dgms[1])
    return float(lt.mean()) if lt.size > 0 else 0.0


@register_metric("persistent_entropy_h0",
                  description="Persistent entropy of H0 diagram")
def persistent_entropy_h0(dgms: list[np.ndarray], **_kw) -> float:
    return persistent_entropy(dgms[0])


@register_metric("persistent_entropy_h1",
                  description="Persistent entropy of H1 diagram")
def persistent_entropy_h1(dgms: list[np.ndarray], **_kw) -> float:
    return persistent_entropy(dgms[1])


@register_metric("h1_l2_norm",
                  description="L2 norm of H1 lifetimes (topological amplitude)")
def h1_l2_norm(dgms: list[np.ndarray], **_kw) -> float:
    lt = _lifetimes(dgms[1])
    return float(np.sqrt(np.sum(lt ** 2))) if lt.size > 0 else 0.0


@register_metric("wasserstein_distance",
                  description="Wasserstein distance between two persistence diagrams")
def wasserstein_distance_metric(
    dgms: list[np.ndarray],
    reference_dgms: list[np.ndarray] = None,
    dim: int = 1,
    **_kw,
) -> float:
    if reference_dgms is None or len(reference_dgms) <= dim:
        return float('nan')
    d1 = _filter_finite(dgms[dim])
    d2 = _filter_finite(reference_dgms[dim])
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return float(persim_wasserstein(d1, d2))


@register_metric("bottleneck_distance",
                  description="Bottleneck distance between two persistence diagrams")
def bottleneck_distance_metric(
    dgms: list[np.ndarray],
    reference_dgms: list[np.ndarray] = None,
    dim: int = 1,
    **_kw,
) -> float:
    if reference_dgms is None or len(reference_dgms) <= dim:
        return float('nan')
    d1 = _filter_finite(dgms[dim])
    d2 = _filter_finite(reference_dgms[dim])
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return float(persim_bottleneck(d1, d2))


# ---------------------------------------------------------------------------
# Takens embedding (local helper for surrogate tests)
# ---------------------------------------------------------------------------

def _takens_embedding(signal: np.ndarray, delay: int, dimension: int) -> np.ndarray:
    """Time-delay (Takens) embedding of a 1-D signal.

    Parameters
    ----------
    signal : 1-D array
    delay : int
        Time delay τ.
    dimension : int
        Embedding dimension m.

    Returns
    -------
    np.ndarray, shape (N - (m-1)*τ, m)
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    rows = n - (dimension - 1) * delay
    if rows <= 0:
        raise ValueError(
            f"Signal length {n} too short for delay={delay}, "
            f"dimension={dimension}."
        )
    indices = np.arange(dimension) * delay
    return np.array([signal[i + indices] for i in range(rows)])


# ---------------------------------------------------------------------------
# AAFT Surrogate Test
# ---------------------------------------------------------------------------

def generate_aaft_surrogate(signal: np.ndarray) -> np.ndarray:
    """Generate an Amplitude-Adjusted Fourier Transform (AAFT) surrogate.

    Algorithm
    ---------
    1. Create a Gaussian random shuffle with the same rank order as *signal*.
    2. Phase-randomise the Gaussian series via FFT.
    3. Re-order the original amplitude values to match the rank order of the
       phase-randomised Gaussian series.

    Parameters
    ----------
    signal : 1-D array

    Returns
    -------
    np.ndarray
        Surrogate time-series with the same amplitude distribution and
        (approximately) the same power spectrum as the original.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)

    # Step 1 — rank-ordered Gaussian series
    sorted_indices = np.argsort(signal)
    gaussian = np.sort(np.random.randn(n))
    rank_gaussian = np.empty(n)
    rank_gaussian[sorted_indices] = gaussian

    # Step 2 — phase-randomise the Gaussian series
    fft_vals = np.fft.rfft(rank_gaussian)
    phases = np.random.uniform(0, 2 * np.pi, len(fft_vals))
    phases[0] = 0  # keep DC component
    if n % 2 == 0:
        phases[-1] = 0  # keep Nyquist component real
    fft_randomised = fft_vals * np.exp(1j * phases)
    randomised_gaussian = np.fft.irfft(fft_randomised, n=n)

    # Step 3 — re-order original amplitudes by surrogate ranks
    surrogate_rank = np.argsort(np.argsort(randomised_gaussian))
    sorted_signal = np.sort(signal)
    surrogate = sorted_signal[surrogate_rank]

    return surrogate


def aaft_surrogate_test(
    signal: np.ndarray,
    delay: int,
    dimension: int,
    n_surrogates: int = 20,
    metric_name: str = "max_h1_persistence",
) -> dict:
    """Non-linear significance test using AAFT surrogates.

    For each surrogate the pipeline is:
    ``signal → Takens embedding → Rips persistence → scalar metric``.

    The real signal goes through the same pipeline and the result is compared
    to the surrogate distribution via a z-score and two-tailed p-value.

    Parameters
    ----------
    signal : 1-D array
        Raw ECG (or other) time-series.
    delay : int
        Time delay τ for Takens embedding.
    dimension : int
        Embedding dimension m.
    n_surrogates : int
        Number of surrogate realisations (default 20).
    metric_name : str
        Name of a registered metric (default ``'max_h1_persistence'``).

    Returns
    -------
    dict
        ``real_value``       — metric computed on the real signal.
        ``surrogate_values`` — list of metric values for each surrogate.
        ``z_score``          — z-score of the real value w.r.t. surrogates.
        ``p_value``          — two-tailed p-value.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    # --- Real signal ---
    pc_real = _takens_embedding(signal, delay, dimension)
    res_real = compute_persistence(pc_real, maxdim=1)
    real_value = compute_metric(metric_name, dgms=res_real["dgms"])

    # --- Surrogates ---
    surrogate_values: list[float] = []
    for _ in range(n_surrogates):
        surr = generate_aaft_surrogate(signal)
        pc_surr = _takens_embedding(surr, delay, dimension)
        res_surr = compute_persistence(pc_surr, maxdim=1)
        sv = compute_metric(metric_name, dgms=res_surr["dgms"])
        surrogate_values.append(float(sv))

    surrogate_arr = np.array(surrogate_values)
    mean_surr = surrogate_arr.mean()
    std_surr = surrogate_arr.std(ddof=1)

    if std_surr == 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = float((real_value - mean_surr) / std_surr)
        p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_score))))

    return {
        "real_value": float(real_value),
        "surrogate_values": surrogate_values,
        "z_score": z_score,
        "p_value": p_value,
    }
