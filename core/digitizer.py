"""
TopoCardio Lab — ECG image digitizer.

Provides two extraction pipelines (column-wise and Viterbi/DP),
calibration helpers, a synthetic ECG generator, and accuracy evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import grey_opening
from scipy.signal import medfilt, savgol_filter
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


# ---------------------------------------------------------------------------
# 3.  Calibration dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationParams:
    """Physical calibration of an ECG paper scan."""

    px_per_mm: float = 10.0
    paper_speed_mm_per_s: float = 25.0
    amplitude_mm_per_mv: float = 10.0

    # derived (computed once on init)
    _px_per_s: float = field(init=False, repr=False)
    _px_per_mv: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._px_per_s = self.px_per_mm * self.paper_speed_mm_per_s
        self._px_per_mv = self.px_per_mm * self.amplitude_mm_per_mv

    def px_to_ms(self, px: float) -> float:
        """Convert a horizontal distance in pixels to milliseconds."""
        return (px / self._px_per_s) * 1000.0

    def px_to_mv(self, px_height: float) -> float:
        """Convert a vertical distance in pixels to millivolts."""
        return px_height / self._px_per_mv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_image(
    filename_or_array: Union[str, NDArray],
) -> Tuple[NDArray, NDArray]:
    """Return (grayscale float H×W, original RGB/gray uint8 H×W×C or H×W).

    Accepts a file path (absolute or relative to BASE_DIR) **or** a numpy
    array that is already loaded.
    """
    if isinstance(filename_or_array, np.ndarray):
        arr = filename_or_array
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3].astype(np.uint8)
            gray = np.array(Image.fromarray(rgb).convert("L"), dtype=float)
        elif arr.ndim == 2:
            rgb = arr.astype(np.uint8)
            gray = arr.astype(float)
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")
        return gray, rgb

    path = filename_or_array
    if not os.path.isabs(path):
        if os.path.exists(path):
            pass
        elif os.path.exists(os.path.join(PROJECT_ROOT, path)):
            path = os.path.join(PROJECT_ROOT, path)
        else:
            path = os.path.join(BASE_DIR, path)

    pil_img = Image.open(path)
    rgb = np.array(pil_img)
    gray = np.array(pil_img.convert("L"), dtype=float)
    return gray, rgb


def _resolve_coords(
    value: float, total: int
) -> int:
    """Interpret *value* as a pixel index (>= 1.0) or a ratio (<= 1.0)."""
    if isinstance(value, (float, int)) and 0.0 <= value <= 1.0:
        return int(round(total * value))
    return int(value)


# ---------------------------------------------------------------------------
# 1.  Column-wise extraction  (original algorithm, full resolution)
# ---------------------------------------------------------------------------

def extract_signal_columnwise(
    filename_or_array: Union[str, NDArray],
    y_start: float = 0.0,
    y_end: float = 1.0,
    x_start: float = 0.05,
    x_end: float = 0.95,
    brightness_threshold: float = 75.0,
    smoothing_window: int = 11,
) -> NDArray:
    """Column-based ECG trace extraction (no downsampling).

    Parameters
    ----------
    filename_or_array : path or ndarray
        Source ECG image.
    y_start, y_end : float
        Vertical crop boundaries (pixel coords ≥ 1 or ratios in (0, 1)).
    x_start, x_end : float
        Horizontal crop boundaries (pixel coords ≥ 1 or ratios in (0, 1)).
    brightness_threshold : float
        Minimum inverted-intensity to treat a pixel as *active*.
    smoothing_window : int
        Window length for Savitzky-Golay smoothing (must be odd).

    Returns
    -------
    signal : 1-D ndarray  (length = cropped width, z-scored and smoothed)
    """
    gray, _ = _load_image(filename_or_array)
    h, w = gray.shape

    r_y0 = _resolve_coords(y_start, h)
    r_y1 = _resolve_coords(y_end, h)
    r_x0 = _resolve_coords(x_start, w)
    r_x1 = _resolve_coords(x_end, w)

    crop = gray[r_y0:r_y1, r_x0:r_x1]
    inv = 255.0 - crop
    inv[inv < brightness_threshold] = 0.0

    crop_h = inv.shape[0]
    signal: list[float] = []

    for col in range(inv.shape[1]):
        col_px = inv[:, col]
        active = np.where(col_px > 0)[0]
        if 0 < len(active) < int(0.35 * crop_h):
            thr = np.percentile(col_px[active], 70)
            y_idx = np.where(col_px >= thr)[0]
            signal.append(-float(np.mean(y_idx)))
        else:
            signal.append(signal[-1] if signal else 0.0)

    sig = np.array(signal, dtype=float)

    # median filter (kernel must be odd)
    sig = medfilt(sig, kernel_size=5)

    # Savitzky-Golay (guard against too-short signals)
    win = smoothing_window if smoothing_window <= len(sig) else (len(sig) // 2) * 2 + 1
    if win >= 5:
        sig = savgol_filter(sig, window_length=win, polyorder=3)

    # z-score
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return sig


# ---------------------------------------------------------------------------
# 2.  Viterbi / Dynamic-Programming extraction
# ---------------------------------------------------------------------------

def _suppress_grid_morphological(inv: NDArray) -> NDArray:
    """Remove coordinate-grid lines from inverted image via morphological opening.

    * Horizontal kernel → detects thin horizontal grid lines.
    * Vertical kernel   → detects thin vertical grid lines.
    Only faint grid lines (< 160 inverted intensity) are suppressed,
    preserving high-intensity ECG peaks (R-waves).
    """
    cleaned = inv.copy()
    h_grid = grey_opening(cleaned, footprint=np.ones((1, 30)))
    v_grid = grey_opening(cleaned, footprint=np.ones((30, 1)))
    grid_mask = (h_grid > 0) | (v_grid > 0)
    cleaned[grid_mask & (cleaned < 160.0)] = 0.0
    return cleaned


def _suppress_grid_color(rgb: NDArray, gray: NDArray) -> NDArray:
    """If the image is RGB, try to remove the coloured grid.

    ECG paper typically has a red/orange or green grid.  We boost the
    *blue-ish* channel (which usually carries the trace ink) and suppress
    grid-coloured pixels.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return gray

    r, g, b = (rgb[:, :, i].astype(float) for i in range(3))

    # Mask pixels that look like red/orange grid
    # (high R, low B, moderate G)
    red_grid = (r > 120) & (b < 100) & (r - b > 60)
    # green grid
    green_grid = (g > 120) & (r < 100) & (b < 100) & (g - r > 50)
    grid_mask = red_grid | green_grid

    out = gray.copy()
    out[grid_mask] = 255.0  # set grid pixels to white (background)
    return out


def extract_signal_viterbi(
    filename_or_array: Union[str, NDArray],
    y_start: float = 0.0,
    y_end: float = 1.0,
    x_start: float = 0.05,
    x_end: float = 0.95,
    continuity_weight: float = 0.01,
    brightness_threshold: float = 50.0,
    smoothing_window: int = 11,
) -> NDArray:
    """Dynamic-programming (Viterbi-style) ECG trace extraction.

    1. Suppress the coordinate grid (colour filtering + morphological opening).
    2. Scan left -> right; at every column consider each row as a possible
       state. The cost of choosing row y at column c is:

           cost = -inv[y, c] + lam * (y - y_prev)^2

       where inv is inverted pixel intensity and lam is continuity_weight.
    3. Backtrack to find the minimum-cost path.
    4. Smooth with median + Savitzky-Golay filters.

    Returns the full native-resolution signal (z-scored).
    """
    gray, rgb = _load_image(filename_or_array)
    h, w = gray.shape

    # --- color grid suppression ---
    cleaned = _suppress_grid_color(rgb, gray)

    # --- crop ---
    r_y0 = _resolve_coords(y_start, h)
    r_y1 = _resolve_coords(y_end, h)
    r_x0 = _resolve_coords(x_start, w)
    r_x1 = _resolve_coords(x_end, w)

    crop = cleaned[r_y0:r_y1, r_x0:r_x1]
    inv = 255.0 - crop
    # --- morphological grid suppression on inverted crop ---
    inv = _suppress_grid_morphological(inv)
    inv[inv < brightness_threshold] = 0.0

    crop_h, crop_w = inv.shape
    lam = continuity_weight

    # --- forward pass (Viterbi) ---
    # cost[y, c] = accumulated minimum cost to reach row y at column c
    cost = np.full((crop_h, crop_w), np.inf, dtype=float)
    back = np.zeros((crop_h, crop_w), dtype=int)

    # initialise first column: cost = -(inverted intensity)
    # We want dark pixels (high inv) ⇒ low cost, so cost = -inv.
    cost[:, 0] = -inv[:, 0]

    for c in range(1, crop_w):
        col_inv = inv[:, c]  # (crop_h,)
        prev_cost = cost[:, c - 1]  # (crop_h,)

        # For efficiency we vectorise over rows.  For each candidate row y
        # we find the best predecessor row y_prev.
        # Total cost = prev_cost[y_prev] + lam*(y - y_prev)^2 - col_inv[y]
        #
        # To keep memory bounded for very tall crops we use a band search
        # (±band rows).  For most ECG images crop_h < 1000 so full search
        # is fine.
        if crop_h <= 1500:
            # full search — simple and exact
            rows = np.arange(crop_h)
            # transition[y, y_prev] = prev_cost[y_prev] + lam*(y-y_prev)^2
            diff = rows[:, None] - rows[None, :]  # (crop_h, crop_h)
            trans = prev_cost[None, :] + lam * (diff ** 2)
            best_prev = np.argmin(trans, axis=1)
            best_trans = trans[rows, best_prev]
        else:
            # banded search for very large images
            band = 150
            best_prev = np.zeros(crop_h, dtype=int)
            best_trans = np.full(crop_h, np.inf)
            for y in range(crop_h):
                lo = max(0, y - band)
                hi = min(crop_h, y + band + 1)
                cands = np.arange(lo, hi)
                vals = prev_cost[lo:hi] + lam * ((y - cands) ** 2)
                idx = np.argmin(vals)
                best_prev[y] = lo + idx
                best_trans[y] = vals[idx]

        cost[:, c] = best_trans - col_inv
        back[:, c] = best_prev

    # --- backtrack ---
    path = np.zeros(crop_w, dtype=int)
    path[-1] = np.argmin(cost[:, -1])
    for c in range(crop_w - 2, -1, -1):
        path[c] = back[path[c + 1], c + 1]

    # invert so that "up on paper" = positive amplitude
    sig = -path.astype(float)

    # --- smooth ---
    sig = medfilt(sig, kernel_size=5)
    win = smoothing_window if smoothing_window <= len(sig) else (len(sig) // 2) * 2 + 1
    if win >= 5:
        sig = savgol_filter(sig, window_length=win, polyorder=3)

    # z-score
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    return sig


# ---------------------------------------------------------------------------
# 4.  Synthetic ECG generator
# ---------------------------------------------------------------------------

def _gaussian(t: NDArray, mu: float, sigma: float, a: float) -> NDArray:
    return a * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _qrs_complex(t: NDArray, centre: float, width: float, amplitude: float) -> NDArray:
    """Sharp triangular QRS approximation."""
    half = width / 2.0
    out = np.zeros_like(t)

    # Q dip (small negative before R)
    q_centre = centre - half * 0.6
    out += _gaussian(t, q_centre, width * 0.08, -0.15 * amplitude)

    # R peak (sharp positive)
    out += _gaussian(t, centre, width * 0.06, amplitude)

    # S dip (small negative after R)
    s_centre = centre + half * 0.6
    out += _gaussian(t, s_centre, width * 0.08, -0.20 * amplitude)

    return out


def _single_beat(
    t: NDArray,
    beat_centre: float,
    rr_interval: float,
    amplitude: float = 1.0,
) -> NDArray:
    """Generate one P-QRS-T beat centred at *beat_centre*."""
    sig = np.zeros_like(t)

    # P wave: ~80 ms before QRS, width ~80 ms
    p_pos = beat_centre - 0.16 * rr_interval
    sig += _gaussian(t, p_pos, 0.035, 0.12 * amplitude)

    # QRS
    sig += _qrs_complex(t, beat_centre, 0.10, amplitude)

    # T wave: ~200 ms after QRS, width ~120 ms
    t_pos = beat_centre + 0.22 * rr_interval
    sig += _gaussian(t, t_pos, 0.055, 0.25 * amplitude)

    return sig


def generate_synthetic_ecg(
    ecg_type: Literal["normal_sinus", "extrasystole", "afib"] = "normal_sinus",
    duration_s: float = 5.0,
    fs: int = 500,
    noise_level: float = 0.02,
    add_grid: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate a synthetic ECG waveform and (optionally) a paper-like image.

    Parameters
    ----------
    ecg_type : str
        ``'normal_sinus'``, ``'extrasystole'``, or ``'afib'``.
    duration_s : float
        Duration in seconds.
    fs : int
        Sampling frequency (Hz).
    noise_level : float
        Std-dev of additive Gaussian noise (relative to QRS amplitude).
    add_grid : bool
        If *True*, render the signal onto an image with 1 mm / 5 mm grid.

    Returns
    -------
    image : ndarray (H, W, 3) uint8 — the rendered image (blank white if
        *add_grid* is False).
    ground_truth : 1-D ndarray — the clean ECG signal.
    time_axis : 1-D ndarray — corresponding time values in seconds.
    """
    rng = np.random.default_rng()
    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = np.zeros(n_samples, dtype=float)

    if ecg_type == "normal_sinus":
        rr = 0.8  # 75 bpm
        beat_times = np.arange(rr / 2, duration_s, rr)
        for bt in beat_times:
            signal += _single_beat(t, bt, rr, amplitude=1.0)

    elif ecg_type == "extrasystole":
        rr = 0.85
        beat_times_list: list[float] = []
        bt = rr / 2
        idx = 0
        while bt < duration_s:
            beat_times_list.append(bt)
            idx += 1
            # insert a premature beat every 4th beat
            if idx % 4 == 3:
                bt += rr * 0.55  # premature
            else:
                bt += rr
        for i, bt in enumerate(beat_times_list):
            amp = 0.70 if (i % 4 == 3) else 1.0
            signal += _single_beat(t, bt, rr, amplitude=amp)

    elif ecg_type == "afib":
        # irregular R-R intervals, no P wave (replace with fibrillatory baseline)
        rr_base = 0.70
        bt = rr_base / 2
        while bt < duration_s:
            rr_cur = rr_base * rng.uniform(0.6, 1.4)
            # QRS + T only (no P)
            signal += _qrs_complex(t, bt, 0.10, 1.0)
            t_pos = bt + 0.22 * rr_cur
            signal += _gaussian(t, t_pos, 0.055, 0.25)
            bt += rr_cur
        # fibrillatory baseline
        fibrillation = 0.04 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        fibrillation += 0.03 * np.sin(2 * np.pi * 9 * t + rng.uniform(0, 2 * np.pi))
        signal += fibrillation
    else:
        raise ValueError(f"Unknown ecg_type: {ecg_type!r}")

    # add noise
    if noise_level > 0:
        signal += rng.normal(0, noise_level, n_samples)

    ground_truth = signal.copy()

    # --- render image ---
    if add_grid:
        image = _render_ecg_image(signal, t, fs)
    else:
        # minimal white image with the trace drawn in black
        image = _render_ecg_image(signal, t, fs, draw_grid=False)

    return image, ground_truth, t


def _render_ecg_image(
    signal: NDArray,
    t: NDArray,
    fs: int,
    draw_grid: bool = True,
    px_per_mm: float = 10.0,
    paper_speed_mm_per_s: float = 25.0,
    amplitude_mm_per_mv: float = 10.0,
) -> NDArray:
    """Render an ECG signal onto a paper-like raster image (uint8 RGB)."""
    duration_s = t[-1] - t[0] + 1.0 / fs
    width_mm = duration_s * paper_speed_mm_per_s
    width_px = int(width_mm * px_per_mm)

    # height: 40 mm (± 20 mm around baseline) is typical for one lead
    height_mm = 40.0
    height_px = int(height_mm * px_per_mm)

    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    if draw_grid:
        # 1 mm fine lines (light)
        fine_step = int(px_per_mm)
        for y in range(0, height_px, max(fine_step, 1)):
            img[y, :] = [255, 200, 200]
        for x in range(0, width_px, max(fine_step, 1)):
            img[:, x] = [255, 200, 200]

        # 5 mm bold lines (darker)
        bold_step = int(5 * px_per_mm)
        for y in range(0, height_px, max(bold_step, 1)):
            img[max(y - 1, 0) : y + 2, :] = [255, 140, 140]
        for x in range(0, width_px, max(bold_step, 1)):
            img[:, max(x - 1, 0) : x + 2] = [255, 140, 140]

    # draw the trace
    baseline_y = height_px // 2
    mv_to_px = px_per_mm * amplitude_mm_per_mv  # px per 1 mV

    # resample signal to width_px
    x_signal = np.linspace(0, len(signal) - 1, width_px)
    resampled = np.interp(x_signal, np.arange(len(signal)), signal)

    for x in range(width_px):
        y_val = baseline_y - int(resampled[x] * mv_to_px)
        y_val = np.clip(y_val, 1, height_px - 2)
        # draw a 3-px wide stroke
        img[y_val - 1 : y_val + 2, x] = [0, 0, 0]

    # connect consecutive columns with a line to avoid gaps
    for x in range(1, width_px):
        y0 = baseline_y - int(resampled[x - 1] * mv_to_px)
        y1 = baseline_y - int(resampled[x] * mv_to_px)
        y0 = int(np.clip(y0, 1, height_px - 2))
        y1 = int(np.clip(y1, 1, height_px - 2))
        lo, hi = min(y0, y1), max(y0, y1)
        img[lo : hi + 1, x] = [0, 0, 0]

    return img


# ---------------------------------------------------------------------------
# 5.  Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_digitization(
    ground_truth: NDArray,
    digitized: NDArray,
) -> Dict[str, float]:
    """Compare a digitised signal against ground truth.

    Both signals are resampled to the same length (the longer of the two)
    and z-scored before comparison.

    Returns
    -------
    dict with keys ``rmse``, ``pearson_r``, ``pearson_p``.
    """
    n = max(len(ground_truth), len(digitized))

    def _resample(sig: NDArray, target_len: int) -> NDArray:
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, sig)

    gt = _resample(ground_truth, n)
    dg = _resample(digitized, n)

    # z-score
    gt = (gt - gt.mean()) / (gt.std() + 1e-8)
    dg = (dg - dg.mean()) / (dg.std() + 1e-8)

    rmse = float(np.sqrt(np.mean((gt - dg) ** 2)))
    r, p = pearsonr(gt, dg)

    return {
        "rmse": rmse,
        "pearson_r": float(r),
        "pearson_p": float(p),
    }


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compat with notebooks / other modules)
# ---------------------------------------------------------------------------

def takens_embedding(
    signal: NDArray, delay: int = 6, dimension: int = 3
) -> NDArray:
    """Time-delay (Takens) embedding of a 1-D signal."""
    n = len(signal) - (dimension - 1) * delay
    cloud = np.zeros((n, dimension))
    for i in range(dimension):
        cloud[:, i] = signal[i * delay : i * delay + n]
    return cloud


def persistent_entropy(dgm: NDArray) -> float:
    """Persistent entropy of a persistence diagram."""
    finite_dgm = dgm[np.isfinite(dgm[:, 1])]
    if len(finite_dgm) == 0:
        return 0.0
    lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    p = lifetimes / np.sum(lifetimes)
    return float(-np.sum(p * np.log(p)))
