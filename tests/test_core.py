"""
TopoCardio Lab — Test Suite for Core Engine Modules.
"""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from core.digitizer import (
    CalibrationParams,
    extract_signal_columnwise,
    extract_signal_viterbi,
    generate_synthetic_ecg,
    evaluate_digitization,
)
from core.embedding import takens_embedding, compute_ami, compute_fnn
from core.tda_engine import (
    compute_persistence,
    get_diagram,
    persistent_entropy,
    get_all_metrics,
    compute_metric,
    generate_aaft_surrogate,
    aaft_surrogate_test,
)
from core.datasets import (
    load_ecg_image,
    ROIPreset,
    save_roi_presets,
    load_roi_presets,
    get_default_ecg_configs,
    get_demo_presets,
    discover_sample_scans,
    load_user_presets,
    save_user_preset,
    resource_path,
)


# ---------------------------------------------------------------------------
# 1. Synthetic ECG & Digitizer Tests
# ---------------------------------------------------------------------------

def test_synthetic_ecg_generation():
    """Verify synthetic ECG generation returns proper arrays and shapes."""
    for ecg_type in ["normal_sinus", "extrasystole", "afib"]:
        img, signal, t = generate_synthetic_ecg(
            ecg_type=ecg_type, duration_s=2.0, fs=250, noise_level=0.01, add_grid=True
        )
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3
        assert len(signal) == len(t)
        assert len(signal) == int(2.0 * 250)
        assert np.isfinite(signal).all()


def test_digitization_accuracy_on_synthetic():
    """Verify extraction accuracy on synthetic ECG with grid."""
    img, gt, t = generate_synthetic_ecg(
        ecg_type="normal_sinus", duration_s=2.0, fs=250, noise_level=0.0, add_grid=True
    )
    
    # Extract using both methods
    sig_v = extract_signal_viterbi(img, 0.0, 1.0, 0.05, 0.95)
    sig_c = extract_signal_columnwise(img, 0.0, 1.0, 0.05, 0.95)
    
    # Crop ground truth to matching region
    start_idx = int(len(gt) * 0.05)
    end_idx = int(len(gt) * 0.95)
    gt_crop = gt[start_idx:end_idx]
    
    eval_v = evaluate_digitization(gt_crop, sig_v)
    eval_c = evaluate_digitization(gt_crop, sig_c)
    
    # Pearson r should be positive and high
    assert eval_v["pearson_r"] > 0.70, f"Viterbi Pearson r {eval_v['pearson_r']} too low"
    assert eval_c["pearson_r"] > 0.70, f"Column-wise Pearson r {eval_c['pearson_r']} too low"
    assert eval_v["rmse"] < 1.0
    assert eval_c["rmse"] < 1.0


def test_calibration_params():
    """Test px to ms and px to mV conversions."""
    cal = CalibrationParams(px_per_mm=10.0, paper_speed_mm_per_s=25.0, amplitude_mm_per_mv=10.0)
    # 250 px = 25 mm = 1.0 second = 1000 ms
    assert np.isclose(cal.px_to_ms(250.0), 1000.0)
    # 100 px = 10 mm = 1.0 mV
    assert np.isclose(cal.px_to_mv(100.0), 1.0)


# ---------------------------------------------------------------------------
# 2. Embedding Tests (Takens, AMI, FNN)
# ---------------------------------------------------------------------------

def test_takens_embedding_shape():
    """Verify Takens embedding produces correct array shape."""
    signal = np.sin(np.linspace(0, 10, 200))
    delay = 5
    dimension = 3
    embedded = takens_embedding(signal, delay=delay, dimension=dimension)
    expected_rows = len(signal) - (dimension - 1) * delay
    assert embedded.shape == (expected_rows, dimension)


def test_takens_embedding_values():
    """Verify exact values of Takens embedding for a simple sequence."""
    signal = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    embedded = takens_embedding(signal, delay=1, dimension=2)
    expected = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ], dtype=float)
    assert np.allclose(embedded, expected)


def test_ami_deterministic():
    """Verify AMI computation is strictly deterministic for fixed inputs."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(300)
    ami1, tau1 = compute_ami(sig, max_lag=20)
    ami2, tau2 = compute_ami(sig, max_lag=20)
    assert np.allclose(ami1, ami2)
    assert tau1 == tau2


def test_ami_finds_minimum():
    """Verify AMI finds a sensible minimum for a periodic sine signal."""
    # Sine wave with 40 samples per period: first minimum is around period/4 = 10
    t = np.linspace(0, 8 * np.pi, 160)
    sig = np.sin(t)
    ami, tau = compute_ami(sig, max_lag=30)
    assert 5 <= tau <= 15


def test_fnn_deterministic():
    """Verify FNN computation is deterministic."""
    t = np.linspace(0, 10 * np.pi, 400)
    sig = np.sin(t)
    fnn1, dim1 = compute_fnn(sig, delay=10, max_dim=5)
    fnn2, dim2 = compute_fnn(sig, delay=10, max_dim=5)
    assert np.allclose(fnn1, fnn2)
    assert dim1 == dim2


def test_fnn_low_dimension_for_simple_signal():
    """A pure 1D sine wave should embed in low dimension (d=2 or d=3)."""
    t = np.linspace(0, 20 * np.pi, 600)
    sig = np.sin(t)
    fnn, opt_dim = compute_fnn(sig, delay=15, max_dim=5)
    # By dimension 2 or 3, FNN fraction should drop sharply
    assert fnn[1] < 0.10 or fnn[2] < 0.10
    assert opt_dim <= 4


# ---------------------------------------------------------------------------
# 3. TDA Engine Tests
# ---------------------------------------------------------------------------

def test_persistence_computation():
    """Circle point cloud must have an evident persistent H1 feature."""
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    res = compute_persistence(circle, maxdim=1)
    assert "dgms" in res
    h1 = res["dgms"][1]
    assert len(h1) > 0
    lifetimes = h1[:, 1] - h1[:, 0]
    assert np.max(lifetimes) > 0.5


def test_metric_registry():
    """Verify all required metrics are registered."""
    metrics = get_all_metrics()
    expected_metrics = [
        "max_h1_persistence",
        "mean_h1_persistence",
        "persistent_entropy_h0",
        "persistent_entropy_h1",
        "h1_l2_norm",
        "wasserstein_distance",
        "bottleneck_distance",
    ]
    for m in expected_metrics:
        assert m in metrics, f"Metric '{m}' not found in registry"


def test_persistent_entropy_nonnegative():
    """Entropy must be non-negative."""
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    res = compute_persistence(circle, maxdim=1)
    e0 = compute_metric("persistent_entropy_h0", dgms=res["dgms"])
    e1 = compute_metric("persistent_entropy_h1", dgms=res["dgms"])
    assert e0 >= 0.0
    assert e1 >= 0.0


def test_tda_scale_invariance_and_proportionality():
    """Scaling point cloud by factor alpha scales persistence by alpha."""
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    alpha = 2.5
    res1 = compute_persistence(circle, maxdim=1)
    res2 = compute_persistence(circle * alpha, maxdim=1)
    
    max_h1_1 = compute_metric("max_h1_persistence", dgms=res1["dgms"])
    max_h1_2 = compute_metric("max_h1_persistence", dgms=res2["dgms"])
    
    assert np.isclose(max_h1_2, max_h1_1 * alpha, rtol=1e-3)
    
    # Entropy is scale-invariant because probabilities p_i = l_i / sum(l) cancel out alpha!
    ent1 = compute_metric("persistent_entropy_h1", dgms=res1["dgms"])
    ent2 = compute_metric("persistent_entropy_h1", dgms=res2["dgms"])
    assert np.isclose(ent1, ent2, rtol=1e-3)


def test_aaft_surrogate_shape():
    """AAFT surrogate preserves signal length and finite properties."""
    t = np.linspace(0, 4 * np.pi, 200)
    sig = np.sin(t) + 0.1 * np.cos(3 * t)
    surr = generate_aaft_surrogate(sig)
    assert len(surr) == len(sig)
    assert np.isfinite(surr).all()
    # Preserves amplitude distribution exactly (sorted values are identical)
    assert np.allclose(np.sort(surr), np.sort(sig))


# ---------------------------------------------------------------------------
# 4. Datasets Tests
# ---------------------------------------------------------------------------

def test_roi_preset_save_load(tmp_path):
    """Test saving and loading ROI presets."""
    preset_file = str(tmp_path / "presets.json")
    presets = [
        ROIPreset(
            name="Test Normal",
            filepath="ЭКГ(1).png",
            x_start=100,
            y_start=200,
            x_end=500,
            y_end=600,
            method="viterbi",
            delay=8,
            dimension=3,
        )
    ]
    save_roi_presets(presets, preset_file)
    loaded = load_roi_presets(preset_file)
    assert len(loaded) == 1
    assert loaded[0].name == "Test Normal"
    assert loaded[0].x_start == 100
    assert loaded[0].method == "viterbi"
    assert loaded[0].delay == 8
    assert loaded[0].dimension == 3


def test_default_ecg_configs():
    """Verify default configurations list."""
    configs = get_default_ecg_configs()
    assert len(configs) >= 3
    for c in configs:
        assert "filename" in c
        assert "label" in c
        assert "y_start_ratio" in c
        assert "x_start_ratio" in c


def test_demo_presets():
    """Verify built-in demo presets."""
    demos = get_demo_presets()
    assert len(demos) == 3
    ids = [d["id"] for d in demos]
    assert "synth_normal" in ids
    assert "synth_extra" in ids
    assert "synth_afib" in ids


def test_user_presets_persistence(tmp_path):
    """Verify user preset saving and loading."""
    custom_file = str(tmp_path / "test_user_presets.json")
    preset_data = {
        "id": "user_test",
        "label": "My Custom ECG",
        "type": "user",
        "filename": "synthetic:normal_sinus",
        "x_start_ratio": 0.1,
        "x_end_ratio": 0.9,
        "y_start_ratio": 0.2,
        "y_end_ratio": 0.5,
        "method": "viterbi",
        "tau": 5,
        "dim": 3,
    }
    save_user_preset(preset_data, filepath=custom_file)
    loaded = load_user_presets(filepath=custom_file)
    assert len(loaded) == 1
    assert loaded[0]["label"] == "My Custom ECG"
    assert loaded[0]["tau"] == 5


def test_discover_sample_scans(tmp_path):
    """Verify discovery of sample scans."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    # Create a small dummy png
    test_img = Image.fromarray(np.full((100, 200), 200, dtype=np.uint8))
    test_img.save(str(samples_dir / "sample_test.png"))

    found = discover_sample_scans(str(samples_dir))
    assert len(found) == 1
    assert found[0]["label"] == "Скан: sample_test.png"
    assert os.path.isfile(found[0]["filename"])


def test_resource_path():
    """Verify resource_path returns valid path."""
    p = resource_path("README.md")
    assert os.path.isabs(p)
    assert os.path.isfile(p)
