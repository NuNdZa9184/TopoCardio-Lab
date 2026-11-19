"""
TopoCardio Lab — Dataset utilities.

Handles ECG image loading, ROI preset persistence, MIT-BIH record
downloading/caching, and default ECG configuration presets.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def resource_path(relative_path: str) -> str:
    """Resolve an absolute path to a resource file for development and PyInstaller bundles."""
    try:
        base_path = sys._MEIPASS  # PyInstaller runtime temporary directory
    except Exception:
        base_path = PROJECT_ROOT

    target = os.path.join(base_path, relative_path)
    if os.path.exists(target):
        return target
    # Fallback to checking from current working directory
    cwd_path = os.path.join(os.getcwd(), relative_path)
    if os.path.exists(cwd_path):
        return cwd_path
    return target


# ---------------------------------------------------------------------------
# 1. Image loading
# ---------------------------------------------------------------------------

def load_ecg_image(filepath: str) -> np.ndarray:
    """Load an ECG image file (.png, .jpg, .bmp) as a grayscale uint8 numpy array.

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        2-D array of shape (height, width), dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file extension is not one of ``.png``, ``.jpg``, ``.jpeg``, ``.bmp``.
    """
    if not os.path.isabs(filepath):
        if os.path.isfile(filepath):
            pass
        elif os.path.isfile(os.path.join(PROJECT_ROOT, filepath)):
            filepath = os.path.join(PROJECT_ROOT, filepath)
        elif os.path.isfile(os.path.join(BASE_DIR, filepath)):
            filepath = os.path.join(BASE_DIR, filepath)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.bmp'):
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            "Expected one of: .png, .jpg, .jpeg, .bmp"
        )

    img = Image.open(filepath).convert('L')  # grayscale
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 2. ROIPreset dataclass
# ---------------------------------------------------------------------------

@dataclass
class ROIPreset:
    """Region-of-interest preset for an ECG image strip.

    Attributes
    ----------
    name : str
        Human-readable label for this preset.
    filepath : str
        Path to the source ECG image.
    x_start, y_start, x_end, y_end : int
        Pixel coordinates of the bounding rectangle (inclusive).
    method : str
        Contour-extraction method — ``'viterbi'`` or ``'columnwise'``.
    delay : int
        Embedding delay (time-delay embedding parameter τ).
    dimension : int
        Embedding dimension (time-delay embedding parameter m).
    """

    name: str
    filepath: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    method: str = 'viterbi'
    delay: int = 1
    dimension: int = 2


# ---------------------------------------------------------------------------
# 3. Save ROI presets to JSON
# ---------------------------------------------------------------------------

def save_roi_presets(presets: List[ROIPreset], filepath: str) -> None:
    """Serialise a list of :class:`ROIPreset` objects to a JSON file.

    Parameters
    ----------
    presets : list[ROIPreset]
        Presets to save.
    filepath : str
        Destination ``.json`` file path.  Parent directories are created
        automatically if they do not exist.
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as fh:
        json.dump(
            [asdict(p) for p in presets],
            fh,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# 4. Load ROI presets from JSON
# ---------------------------------------------------------------------------

def load_roi_presets(filepath: str) -> List[ROIPreset]:
    """Deserialise a list of :class:`ROIPreset` objects from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the ``.json`` file previously written by
        :func:`save_roi_presets`.

    Returns
    -------
    list[ROIPreset]
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ROI presets file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as fh:
        raw: list = json.load(fh)

    return [ROIPreset(**item) for item in raw]


# ---------------------------------------------------------------------------
# 5. MIT-BIH record loading via wfdb
# ---------------------------------------------------------------------------

def load_mitbih_record(
    record_name: str,
    cache_dir: str = 'data/mitbih',
    channel: int = 0,
) -> Tuple[np.ndarray, float]:
    """Download (or read from cache) a MIT-BIH Arrhythmia Database record.

    Uses the `wfdb <https://pypi.org/project/wfdb/>`_ library to fetch the
    record from PhysioNet and cache it locally.

    Parameters
    ----------
    record_name : str
        Record identifier, e.g. ``'100'``, ``'101'``, …
    cache_dir : str
        Local directory where downloaded ``.dat`` / ``.hea`` files are stored.
    channel : int
        Which signal channel to return (default ``0``).

    Returns
    -------
    signal : np.ndarray
        1-D float64 array of the physical signal values (mV).
    fs : float
        Sampling frequency in Hz.

    Raises
    ------
    RuntimeError
        On network errors or when the record cannot be loaded.
    """
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'wfdb' package is required for MIT-BIH support. "
            "Install it with:  pip install wfdb"
        ) from exc

    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, record_name)

    # Check whether the record is already cached (both .hea and .dat present)
    hea_exists = os.path.isfile(local_path + '.hea')
    dat_exists = os.path.isfile(local_path + '.dat')

    if hea_exists and dat_exists:
        try:
            record = wfdb.rdrecord(local_path)
            signal = record.p_signal[:, channel].astype(np.float64)
            return signal, float(record.fs)
        except Exception:
            # Cached files may be corrupted — fall through to re-download.
            pass

    # Download from PhysioNet
    try:
        record = wfdb.rdrecord(
            record_name,
            pn_dir='mitdb',
        )
        # Cache to disk
        try:
            wfdb.rdrecord(record_name, pn_dir='mitdb')
            # wfdb.dl_database can download individual records; alternatively
            # we can simply use wfdb.io.dl_files for caching.  A lightweight
            # approach: download header + data files explicitly.
            _physionet_base = 'https://physionet.org/files/mitdb/1.0.0/'
            wfdb.dl_files(
                db='mitdb',
                dl_dir=cache_dir,
                files=[f'{record_name}.hea', f'{record_name}.dat'],
            )
        except Exception:
            # Caching failure is non-fatal.
            pass

        signal = record.p_signal[:, channel].astype(np.float64)
        return signal, float(record.fs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MIT-BIH record '{record_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 6. List standard MIT-BIH record names
# ---------------------------------------------------------------------------

_MITBIH_RECORDS: List[str] = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219',
    '220', '221', '222', '223', '228', '230', '231', '232', '233', '234',
]


def list_mitbih_records() -> List[str]:
    """Return the list of standard MIT-BIH Arrhythmia Database record names.

    Returns
    -------
    list[str]
        48 record identifiers, e.g. ``['100', '101', …, '234']``.
    """
    return list(_MITBIH_RECORDS)


# ---------------------------------------------------------------------------
# 7. Demo presets, sample scanner & user presets persistence
# ---------------------------------------------------------------------------

def get_demo_presets() -> List[dict]:
    """Return built-in demo presets generated synthetically (no external image required)."""
    return [
        {
            "id": "synth_normal",
            "filename": "synthetic:normal_sinus",
            "label": "Демо: Синусовый ритм (Норма)",
            "type": "synthetic",
            "ecg_type": "normal_sinus",
            "y_start_ratio": 0.15,
            "y_end_ratio": 0.85,
            "x_start_ratio": 0.05,
            "x_end_ratio": 0.95,
        },
        {
            "id": "synth_extra",
            "filename": "synthetic:extrasystole",
            "label": "Демо: Экстрасистолия",
            "type": "synthetic",
            "ecg_type": "extrasystole",
            "y_start_ratio": 0.15,
            "y_end_ratio": 0.85,
            "x_start_ratio": 0.05,
            "x_end_ratio": 0.95,
        },
        {
            "id": "synth_afib",
            "filename": "synthetic:afib",
            "label": "Демо: Фибрилляция предсердий",
            "type": "synthetic",
            "ecg_type": "afib",
            "y_start_ratio": 0.15,
            "y_end_ratio": 0.85,
            "x_start_ratio": 0.05,
            "x_end_ratio": 0.95,
        },
    ]


def discover_sample_scans(samples_dir: str = "samples") -> List[dict]:
    """Scan samples_dir for image files if directory exists, returning list of configs."""
    resolved_dir = resource_path(samples_dir)
    if not os.path.isdir(resolved_dir):
        if not os.path.isdir(samples_dir):
            return []
        resolved_dir = os.path.abspath(samples_dir)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    scans = []
    try:
        for fname in sorted(os.listdir(resolved_dir)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                full_path = os.path.join(resolved_dir, fname)
                scans.append({
                    "id": f"sample_{fname}",
                    "filename": full_path,
                    "label": f"Скан: {fname}",
                    "type": "file",
                    "y_start_ratio": 0.20,
                    "y_end_ratio": 0.50,
                    "x_start_ratio": 0.05,
                    "x_end_ratio": 0.95,
                })
    except Exception:
        return []
    return scans


def load_user_presets(filepath: str = "user_presets.json") -> List[dict]:
    """Load user-defined presets from JSON file."""
    if not os.path.isabs(filepath):
        filepath = resource_path(filepath)
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_user_preset(preset_dict: dict, filepath: str = "user_presets.json") -> None:
    """Save or update a user preset in JSON file."""
    if not os.path.isabs(filepath):
        filepath = resource_path(filepath)
    existing = load_user_presets(filepath)
    # Deduplicate by label
    label = preset_dict.get("label", "")
    existing = [p for p in existing if p.get("label") != label]
    existing.append(preset_dict)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def get_default_ecg_configs() -> List[dict]:
    """Return universal default presets list (synthetic demos + any discovered scans).

    Guaranteed to return at least 3 configs for compatibility with test assertions.
    """
    presets = get_demo_presets()
    samples = discover_sample_scans()
    return presets + samples
