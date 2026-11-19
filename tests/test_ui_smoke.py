"""
Smoke test for TopoCardio Lab UI and export functions.
Runs headless off-screen test verifying that all tabs and export routines execute without errors.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.export_utils import export_figure_publication, export_table_csv, export_table_xlsx


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication(sys.argv)
    return app


def test_main_window_init(qapp):
    """Test window creation, tabs existence, and initial state."""
    window = MainWindow()
    assert window.tabs.count() == 3
    assert window.tabs.tabText(0).startswith("1.")
    assert window.tabs.tabText(1).startswith("2.")
    assert window.tabs.tabText(2).startswith("3.")
    # Process initial sample loading
    window._load_initial_sample()
    qapp.processEvents()
    assert window.current_image_array is not None
    assert window.current_image_path is not None
    window.thread_pool.waitForDone(2000)
    window.close()


def test_export_utilities(tmp_path):
    """Test figure export (PNG 300 DPI + SVG) and table export (CSV + XLSX)."""
    # Sample data
    t = np.linspace(0, 4 * np.pi, 200)
    sig = np.sin(t)
    cloud = np.column_stack([sig[:-2], sig[1:-1], sig[2:]])
    dgms = [np.array([[0.0, 1.0], [0.0, 0.5]]), np.array([[0.2, 0.8]])]
    metrics = {
        "max_h1_persistence": 0.6,
        "mean_h1_persistence": 0.6,
        "persistent_entropy_h0": 0.693,
        "persistent_entropy_h1": 0.0,
        "h1_l2_norm": 0.6,
        "wasserstein_distance": 0.0,
        "bottleneck_distance": 0.0,
    }

    # 1. Figure export
    base_fig = str(tmp_path / "test_report_fig")
    res_fig = export_figure_publication(sig, cloud, dgms, metrics, base_fig, dpi=300)
    assert os.path.isfile(res_fig["png"])
    assert os.path.isfile(res_fig["svg"])
    assert os.path.getsize(res_fig["png"]) > 10000
    assert os.path.getsize(res_fig["svg"]) > 1000

    # 2. Table export
    data_rows = [
        {
            "Источник": "ЭКГ(1).png",
            "Участок (ROI)": "Норма",
            "Метод": "Витерби",
            "τ": 6,
            "d": 3,
            "Max H1": 0.8543,
            "Mean H1": 0.1234,
            "Энтропия H0": 5.4321,
            "Энтропия H1": 3.1415,
            "H1 L2-норма": 1.2345,
            "Вассерштейн": 0.0,
            "Боттлнек": 0.0,
        }
    ]

    csv_path = str(tmp_path / "test_table.csv")
    export_table_csv(data_rows, csv_path)
    assert os.path.isfile(csv_path)
    assert os.path.getsize(csv_path) > 50

    xlsx_path = str(tmp_path / "test_table.xlsx")
    export_table_xlsx(data_rows, xlsx_path)
    assert os.path.isfile(xlsx_path)
    assert os.path.getsize(xlsx_path) > 1000


def test_batch_table_operations(qapp):
    """Test batch table insertion and clearing."""
    window = MainWindow()
    row = {
        "Источник": "test_ecg.png",
        "Участок (ROI)": "ROI #1",
        "Метод": "Витерби",
        "τ": 6,
        "d": 3,
        "Max H1": 0.5,
        "Mean H1": 0.2,
        "Энтропия H0": 1.0,
        "Энтропия H1": 0.5,
        "H1 L2-норма": 0.3,
        "Вассерштейн": 0.0,
        "Боттлнек": 0.0,
    }
    window.batch_dataset_rows.append(row)
    window._insert_row_into_batch_table(row)
    assert window.batch_table.rowCount() == 1
    assert window.batch_table.item(0, 0).text() == "test_ecg.png"

    window._on_clear_batch_table()
    assert window.batch_table.rowCount() == 0
    assert len(window.batch_dataset_rows) == 0
    window.close()
