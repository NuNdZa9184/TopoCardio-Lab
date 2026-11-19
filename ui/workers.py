"""
TopoCardio Lab — Asynchronous Background Workers for UI.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, Signal, Slot

from core.digitizer import extract_signal_columnwise, extract_signal_viterbi
from core.embedding import compute_ami, compute_fnn, takens_embedding
from core.tda_engine import (
    aaft_surrogate_test,
    compute_metric,
    compute_persistence,
    get_all_metrics,
)


class WorkerSignals(QObject):
    """Signals for background workers."""
    started = Signal()
    progress = Signal(int, str)  # percentage, status message
    finished = Signal(object)    # result object
    error = Signal(str)          # error message traceback


class DigitizationWorker(QRunnable):
    """Background worker for ECG trace extraction."""

    def __init__(
        self,
        image_or_path: Any,
        y_start: float,
        y_end: float,
        x_start: float,
        x_end: float,
        method: str = "viterbi",
        brightness_threshold: float = 50.0,
        smoothing_window: int = 11,
        continuity_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.image_or_path = image_or_path
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end
        self.method = method
        self.brightness_threshold = brightness_threshold
        self.smoothing_window = smoothing_window
        self.continuity_weight = continuity_weight

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            if self.method == "columnwise":
                sig = extract_signal_columnwise(
                    self.image_or_path,
                    y_start=self.y_start,
                    y_end=self.y_end,
                    x_start=self.x_start,
                    x_end=self.x_end,
                    brightness_threshold=self.brightness_threshold,
                    smoothing_window=self.smoothing_window,
                )
            else:
                sig = extract_signal_viterbi(
                    self.image_or_path,
                    y_start=self.y_start,
                    y_end=self.y_end,
                    x_start=self.x_start,
                    x_end=self.x_end,
                    continuity_weight=self.continuity_weight,
                    brightness_threshold=self.brightness_threshold,
                    smoothing_window=self.smoothing_window,
                )
            self.signals.finished.emit(sig)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())


class AnalysisWorker(QRunnable):
    """Background worker for Takens embedding and TDA persistence calculation."""

    def __init__(
        self,
        signal: np.ndarray,
        delay: int = 6,
        dimension: int = 3,
        reference_dgms: Optional[List[np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.signal = signal
        self.delay = delay
        self.dimension = dimension
        self.reference_dgms = reference_dgms

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            # 1. Takens embedding
            cloud = takens_embedding(self.signal, delay=self.delay, dimension=self.dimension)
            
            # 2. Ripser persistence
            res = compute_persistence(cloud, maxdim=1)
            dgms = res["dgms"]
            
            # 3. Compute all registered metrics
            metrics_dict = {}
            for name, meta in get_all_metrics().items():
                try:
                    val = compute_metric(
                        name, dgms=dgms, reference_dgms=self.reference_dgms, dim=1
                    )
                    metrics_dict[name] = val
                except Exception:
                    metrics_dict[name] = float("nan")

            result = {
                "cloud": cloud,
                "dgms": dgms,
                "metrics": metrics_dict,
                "delay": self.delay,
                "dimension": self.dimension,
            }
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())


class ParameterEstimationWorker(QRunnable):
    """Background worker for AMI and FNN computation."""

    def __init__(self, signal: np.ndarray, max_lag: int = 40, max_dim: int = 6) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.signal = signal
        self.max_lag = max_lag
        self.max_dim = max_dim

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            # AMI
            ami_vals, optimal_tau = compute_ami(self.signal, max_lag=self.max_lag)
            # FNN with optimal tau
            fnn_vals, optimal_d = compute_fnn(self.signal, delay=optimal_tau, max_dim=self.max_dim)
            
            result = {
                "ami_vals": ami_vals,
                "optimal_tau": optimal_tau,
                "fnn_vals": fnn_vals,
                "optimal_d": optimal_d,
            }
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())


class SurrogateWorker(QRunnable):
    """Background worker for AAFT surrogate test."""

    def __init__(
        self,
        signal: np.ndarray,
        delay: int,
        dimension: int,
        n_surrogates: int = 20,
        metric_name: str = "max_h1_persistence",
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.signal = signal
        self.delay = delay
        self.dimension = dimension
        self.n_surrogates = n_surrogates
        self.metric_name = metric_name

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            res = aaft_surrogate_test(
                signal=self.signal,
                delay=self.delay,
                dimension=self.dimension,
                n_surrogates=self.n_surrogates,
                metric_name=self.metric_name,
            )
            self.signals.finished.emit(res)
        except Exception as exc:
            self.signals.error.emit(traceback.format_exc())
