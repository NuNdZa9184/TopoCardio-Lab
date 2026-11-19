"""
TopoCardio Lab — Main Application Window.
Desktop Scientific Workstation for Topological Data Analysis of ECG signals.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.opengl import GLScatterPlotItem, GLViewWidget
from PySide6.QtCore import QThreadPool, QTimer, Qt, Slot
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.datasets import (
    ROIPreset,
    discover_sample_scans,
    get_default_ecg_configs,
    get_demo_presets,
    list_mitbih_records,
    load_ecg_image,
    load_mitbih_record,
    load_roi_presets,
    load_user_presets,
    resource_path,
    save_roi_presets,
    save_user_preset,
)
from core.digitizer import (
    CalibrationParams,
    evaluate_digitization,
    generate_synthetic_ecg,
)
from core.embedding import compute_ami, compute_fnn, takens_embedding
from core.tda_engine import (
    compute_metric,
    compute_persistence,
    get_all_metrics,
)
from ui.export_utils import (
    export_figure_publication,
    export_table_csv,
    export_table_xlsx,
)
from ui.workers import (
    AnalysisWorker,
    DigitizationWorker,
    ParameterEstimationWorker,
    SurrogateWorker,
)

# Enable antialiasing in pyqtgraph
pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")


class MainWindow(QMainWindow):
    """Main window of TopoCardio Lab."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TopoCardio Lab — Топологический анализ ЭКГ (TDA)")
        self.resize(1400, 900)

        # Set Window Icon from TopoCardio_Lab_icon.svg if present
        icon_path = resource_path("TopoCardio_Lab_icon.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.thread_pool = QThreadPool.globalInstance()

        # State
        self.current_image_path: Optional[str] = None
        self.current_image_array: Optional[np.ndarray] = None
        self.current_signal: Optional[np.ndarray] = None
        self.current_cloud: Optional[np.ndarray] = None
        self.current_dgms: Optional[List[np.ndarray]] = None
        self.current_metrics: Dict[str, float] = {}

        # Reference standard (Norm) for distance metrics
        self.reference_dgms: Optional[List[np.ndarray]] = None
        self.reference_label: str = "Не задан"

        # Calibration
        self.calibration = CalibrationParams()

        # Multiple ROIs management
        self.rois: List[pg.RectROI] = []
        self.roi_colors = ["#ff3333", "#33cc33", "#3388ff", "#ffaa00", "#cc33ff"]
        self.active_roi_index: int = 0

        # Batch table state
        self.batch_selected_files: List[str] = []
        self.batch_dataset_rows: List[Dict[str, Any]] = []

        # 3D view rotation timer
        self.rotation_timer = QTimer(self)
        self.rotation_timer.timeout.connect(self._rotate_3d_view)
        self.current_azimuth = 45.0

        self._build_ui()
        self._apply_styling()

        # Load initial default sample
        QTimer.singleShot(100, self._load_initial_sample)

    # =========================================================================
    # UI Construction
    # =========================================================================

    def _build_ui(self) -> None:
        """Create tabs and status bar."""
        # Main Tab Widget
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Tab 1: Interactive Scan & ROI Analysis
        self.tab_analysis = QWidget()
        self._build_tab_analysis(self.tab_analysis)
        self.tabs.addTab(self.tab_analysis, "1. Интерактивный анализ (ROI)")

        # Tab 2: Parameter Justification & Validation
        self.tab_validation = QWidget()
        self._build_tab_validation(self.tab_validation)
        self.tabs.addTab(self.tab_validation, "2. Обоснование параметров (AMI / FNN / AAFT)")

        # Tab 3: Batch Experiment & Export
        self.tab_batch = QWidget()
        self._build_tab_batch(self.tab_batch)
        self.tabs.addTab(self.tab_batch, "3. Пакетный эксперимент и экспорт")

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Готов к работе")

    # -------------------------------------------------------------------------
    # TAB 1: Interactive Analysis
    # -------------------------------------------------------------------------
    def _build_tab_analysis(self, parent: QWidget) -> None:
        main_layout = QHBoxLayout(parent)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Splitter dividing Left (Scan + Controls) and Right (Signals + TDA)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT PANEL: Controls & Image with ROI
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # 1. Top Controls Bar (Open, Presets, MIT-BIH)
        top_ctrl_group = QGroupBox("Источник данных")
        top_ctrl_layout = QHBoxLayout(top_ctrl_group)
        top_ctrl_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_open_scan = QPushButton("Открыть скан...")
        self.btn_open_scan.clicked.connect(self._on_open_scan_dialog)
        top_ctrl_layout.addWidget(self.btn_open_scan)

        self.combo_presets = QComboBox()
        self._populate_presets_combo()
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        top_ctrl_layout.addWidget(self.combo_presets)

        self.btn_save_preset = QPushButton("Сохранить пресет...")
        self.btn_save_preset.clicked.connect(self._on_save_preset_dialog)
        top_ctrl_layout.addWidget(self.btn_save_preset)

        self.btn_mitbih = QPushButton("База PhysioNet MIT-BIH...")
        self.btn_mitbih.clicked.connect(self._on_mitbih_dialog)
        top_ctrl_layout.addWidget(self.btn_mitbih)

        self.btn_synth = QPushButton("Синтез ЭКГ...")
        self.btn_synth.clicked.connect(self._on_synth_dialog)
        top_ctrl_layout.addWidget(self.btn_synth)

        left_layout.addWidget(top_ctrl_group)

        # 2. Image View with ROI
        img_group = QGroupBox("Скан ЭКГ с рамкой выделения (ROI)")
        img_layout = QVBoxLayout(img_group)
        img_layout.setContentsMargins(4, 4, 4, 4)

        self.image_plot_widget = pg.PlotWidget()
        self.image_plot_widget.setLabel("bottom", "Пиксели (X)")
        self.image_plot_widget.setLabel("left", "Пиксели (Y)")
        self.image_plot_widget.invertY(True)  # Standard image coordinates (row 0 at top)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_plot_widget.addItem(self.image_item)
        img_layout.addWidget(self.image_plot_widget)

        # ROI control buttons
        roi_btns_layout = QHBoxLayout()
        self.btn_add_roi = QPushButton("Добавить ROI")
        self.btn_add_roi.clicked.connect(self._on_add_roi)
        roi_btns_layout.addWidget(self.btn_add_roi)

        self.btn_del_roi = QPushButton("Удалить активный ROI")
        self.btn_del_roi.clicked.connect(self._on_delete_active_roi)
        roi_btns_layout.addWidget(self.btn_del_roi)

        self.btn_set_reference = QPushButton("Зафиксировать как эталон (Норма)")
        self.btn_set_reference.clicked.connect(self._on_set_reference)
        roi_btns_layout.addWidget(self.btn_set_reference)

        self.lbl_reference_status = QLabel("Эталон: Не задан")
        self.lbl_reference_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        roi_btns_layout.addWidget(self.lbl_reference_status)

        img_layout.addLayout(roi_btns_layout)
        left_layout.addWidget(img_group, stretch=1)

        # 3. Extraction & Calibration Parameters
        params_group = QGroupBox("Параметры оцифровки и вложения")
        params_layout = QHBoxLayout(params_group)
        params_layout.setContentsMargins(6, 6, 6, 6)

        # Column 1: Extraction
        form1 = QFormLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Витерби (DP)", "Поколоночный"])
        self.combo_method.currentIndexChanged.connect(self._trigger_reanalysis)
        form1.addRow("Метод:", self.combo_method)

        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(5, 240)
        self.spin_thresh.setValue(50)
        self.spin_thresh.valueChanged.connect(self._trigger_reanalysis)
        form1.addRow("Порог яркости:", self.spin_thresh)

        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(3, 51)
        self.spin_smooth.setSingleStep(2)
        self.spin_smooth.setValue(11)
        self.spin_smooth.valueChanged.connect(self._trigger_reanalysis)
        form1.addRow("Окно фильтра:", self.spin_smooth)
        params_layout.addLayout(form1)

        # Column 2: Embedding
        form2 = QFormLayout()
        h_tau = QHBoxLayout()
        self.spin_tau = QSpinBox()
        self.spin_tau.setRange(1, 100)
        self.spin_tau.setValue(6)
        self.spin_tau.valueChanged.connect(self._trigger_reembedding)
        h_tau.addWidget(self.spin_tau)

        self.btn_auto_tau = QPushButton("Авто (AMI)")
        self.btn_auto_tau.clicked.connect(self._on_auto_tau_click)
        h_tau.addWidget(self.btn_auto_tau)
        form2.addRow("Задержка (τ):", h_tau)

        self.spin_dim = QSpinBox()
        self.spin_dim.setRange(2, 6)
        self.spin_dim.setValue(3)
        self.spin_dim.valueChanged.connect(self._trigger_reembedding)
        form2.addRow("Размерность (d):", self.spin_dim)

        self.spin_px_per_mm = QDoubleSpinBox()
        self.spin_px_per_mm.setRange(1.0, 100.0)
        self.spin_px_per_mm.setValue(10.0)
        self.spin_px_per_mm.valueChanged.connect(self._on_calibration_changed)
        form2.addRow("Пикс / мм:", self.spin_px_per_mm)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["25 мм/с", "50 мм/с"])
        self.combo_speed.currentIndexChanged.connect(self._on_calibration_changed)
        form2.addRow("Скорость:", self.combo_speed)
        params_layout.addLayout(form2)

        left_layout.addWidget(params_group)

        splitter.addWidget(left_widget)

        # RIGHT PANEL: Visualizations (Signal, 3D Attractor, Persistence) & Metrics
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        # Top Right: Signal Plot
        sig_group = QGroupBox("Оцифрованный сигнал x(t)")
        sig_layout = QVBoxLayout(sig_group)
        sig_layout.setContentsMargins(4, 4, 4, 4)
        self.signal_plot_widget = pg.PlotWidget()
        self.signal_plot_widget.setLabel("bottom", "Время (мс)")
        self.signal_plot_widget.setLabel("left", "Амплитуда (Z-score)")
        self.signal_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.signal_curve = self.signal_plot_widget.plot(pen=pg.mkPen("#00ccff", width=1.5))
        sig_layout.addWidget(self.signal_plot_widget)
        right_layout.addWidget(sig_group, stretch=1)

        # Middle Right: Split 3D Attractor and Persistence Diagram
        mid_splitter = QSplitter(Qt.Horizontal)

        # 3D Attractor
        attractor_group = QGroupBox("Аттрактор Такенса (x(t), x(t+τ), x(t+2τ))")
        attractor_layout = QVBoxLayout(attractor_group)
        attractor_layout.setContentsMargins(4, 4, 4, 4)

        try:
            self.gl_view = GLViewWidget()
            self.gl_view.setCameraPosition(distance=10, elevation=25, azimuth=45)
            self.gl_scatter = GLScatterPlotItem()
            self.gl_view.addItem(self.gl_scatter)
            attractor_layout.addWidget(self.gl_view)
            self.has_opengl = True
        except Exception:
            self.has_opengl = False
            self.gl_fallback_plot = pg.PlotWidget()
            self.gl_fallback_plot.setLabel("bottom", "x(t)")
            self.gl_fallback_plot.setLabel("left", "x(t+τ)")
            attractor_layout.addWidget(self.gl_fallback_plot)

        # Auto-rotation checkbox
        h_rot = QHBoxLayout()
        self.chk_rotate = QCheckBox("Вращение 3D")
        self.chk_rotate.toggled.connect(self._toggle_rotation)
        h_rot.addWidget(self.chk_rotate)
        attractor_layout.addLayout(h_rot)

        mid_splitter.addWidget(attractor_group)

        # Persistence Diagram
        dgm_group = QGroupBox("Диаграмма персистентности (Ripser)")
        dgm_layout = QVBoxLayout(dgm_group)
        dgm_layout.setContentsMargins(4, 4, 4, 4)
        self.dgm_plot_widget = pg.PlotWidget()
        self.dgm_plot_widget.setLabel("bottom", "Рождение (Birth)")
        self.dgm_plot_widget.setLabel("left", "Смерть (Death)")
        self.dgm_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.dgm_plot_widget.addLegend(offset=(10, 10))

        # Identity diagonal
        self.dgm_diagonal = self.dgm_plot_widget.plot(pen=pg.mkPen("#888888", style=Qt.DashLine))
        self.dgm_h0_scatter = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen(None), brush=pg.mkBrush("#3388ff"), name="H0 (компоненты)"
        )
        self.dgm_h1_scatter = pg.ScatterPlotItem(
            size=9, symbol="t", pen=pg.mkPen(None), brush=pg.mkBrush("#ff3333"), name="H1 (циклы)"
        )
        self.dgm_plot_widget.addItem(self.dgm_h0_scatter)
        self.dgm_plot_widget.addItem(self.dgm_h1_scatter)
        dgm_layout.addWidget(self.dgm_plot_widget)

        mid_splitter.addWidget(dgm_group)
        right_layout.addWidget(mid_splitter, stretch=2)

        # Bottom Right: Metrics Table
        metrics_group = QGroupBox("Вычисленные топологические метрики (TDA)")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setContentsMargins(4, 4, 4, 4)

        self.table_metrics = QTableWidget(1, 7)
        self.table_metrics.setHorizontalHeaderLabels([
            "Max H1", "Mean H1", "Энтропия H0", "Энтропия H1",
            "H1 L2-норма", "Вассерштейн (к Норме)", "Боттлнек (к Норме)"
        ])
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_metrics.verticalHeader().setVisible(False)
        self.table_metrics.setFixedHeight(65)
        metrics_layout.addWidget(self.table_metrics)

        # Button to add to batch
        h_batch_add = QHBoxLayout()
        self.btn_add_to_batch = QPushButton("Добавить участок в сводную таблицу")
        self.btn_add_to_batch.setStyleSheet("background-color: #24527a; font-weight: 600; padding: 7px 14px;")
        self.btn_add_to_batch.clicked.connect(self._on_add_to_batch_table)
        h_batch_add.addWidget(self.btn_add_to_batch)
        metrics_layout.addLayout(h_batch_add)

        right_layout.addWidget(metrics_group)

        splitter.addWidget(right_widget)
        splitter.setSizes([650, 750])

    # -------------------------------------------------------------------------
    # TAB 2: Parameter Justification & Validation
    # -------------------------------------------------------------------------
    def _build_tab_validation(self, parent: QWidget) -> None:
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top row: AMI and FNN plots
        top_splitter = QSplitter(Qt.Horizontal)

        # AMI Plot
        ami_group = QGroupBox("1. Взаимная информация AMI(τ) — выбор оптимальной задержки")
        ami_layout = QVBoxLayout(ami_group)
        self.ami_plot = pg.PlotWidget()
        self.ami_plot.setLabel("bottom", "Задержка τ (отсчёты)")
        self.ami_plot.setLabel("left", "AMI (бит)")
        self.ami_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ami_curve = self.ami_plot.plot(pen=pg.mkPen("#00ffcc", width=2))
        self.ami_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff3333", width=2, style=Qt.DashLine))
        self.ami_plot.addItem(self.ami_marker)
        self.lbl_ami_opt = QLabel("Оптимальный τ (1-й локальный минимум): —")
        self.lbl_ami_opt.setStyleSheet("font-weight: bold; color: #00ffcc;")
        ami_layout.addWidget(self.ami_plot)
        ami_layout.addWidget(self.lbl_ami_opt)
        top_splitter.addWidget(ami_group)

        # FNN Plot
        fnn_group = QGroupBox("2. Метод ложных соседей FNN(d) — обоснование размерности")
        fnn_layout = QVBoxLayout(fnn_group)
        self.fnn_plot = pg.PlotWidget()
        self.fnn_plot.setLabel("bottom", "Размерность d")
        self.fnn_plot.setLabel("left", "Доля FNN (%)")
        self.fnn_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fnn_curve = self.fnn_plot.plot(pen=pg.mkPen("#ff9933", width=2), symbol="o", symbolSize=8)
        self.lbl_fnn_opt = QLabel("Оптимальная размерность вложения d: —")
        self.lbl_fnn_opt.setStyleSheet("font-weight: bold; color: #ff9933;")
        fnn_layout.addWidget(self.fnn_plot)
        fnn_layout.addWidget(self.lbl_fnn_opt)
        top_splitter.addWidget(fnn_group)

        main_layout.addWidget(top_splitter, stretch=1)

        # Bottom row: AAFT Surrogate test
        surr_group = QGroupBox("3. Тест суррогатных данных (AAFT) — проверка нелинейности динамики")
        surr_layout = QVBoxLayout(surr_group)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Число суррогатов N:"))
        self.spin_surr_n = QSpinBox()
        self.spin_surr_n.setRange(5, 100)
        self.spin_surr_n.setValue(20)
        ctrl_layout.addWidget(self.spin_surr_n)

        ctrl_layout.addWidget(QLabel("Метрика:"))
        self.combo_surr_metric = QComboBox()
        self.combo_surr_metric.addItems([
            "max_h1_persistence", "h1_l2_norm", "mean_h1_persistence", "persistent_entropy_h1"
        ])
        ctrl_layout.addWidget(self.combo_surr_metric)

        self.btn_run_surr = QPushButton("Запустить тест суррогатов")
        self.btn_run_surr.setStyleSheet("background-color: #1b633b; font-weight: 600; padding: 7px 14px;")
        self.btn_run_surr.clicked.connect(self._on_run_surrogate_test)
        ctrl_layout.addWidget(self.btn_run_surr)
        ctrl_layout.addStretch()
        surr_layout.addLayout(ctrl_layout)

        # Surrogate distribution plot + stats
        surr_split = QSplitter(Qt.Horizontal)
        self.surr_plot = pg.PlotWidget()
        self.surr_plot.setLabel("bottom", "Значение метрики")
        self.surr_plot.setLabel("left", "Частота / Номер")
        self.surr_plot.showGrid(x=True, y=True, alpha=0.3)
        self.surr_real_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff0000", width=2.5))
        self.surr_plot.addItem(self.surr_real_line)
        surr_split.addWidget(self.surr_plot)

        # Stats info box
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.lbl_surr_real = QLabel("Реальный сигнал: —")
        self.lbl_surr_mean = QLabel("Суррогаты (Mean ± Std): —")
        self.lbl_surr_z = QLabel("Z-оценка (z-score): —")
        self.lbl_surr_p = QLabel("p-value: —")
        self.lbl_surr_verdict = QLabel("Заключение: запустите тест")
        self.lbl_surr_verdict.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffcc00;")
        for w in [self.lbl_surr_real, self.lbl_surr_mean, self.lbl_surr_z, self.lbl_surr_p, self.lbl_surr_verdict]:
            stats_layout.addWidget(w)
        stats_layout.addStretch()
        surr_split.addWidget(stats_widget)

        surr_layout.addWidget(surr_split)
        main_layout.addWidget(surr_group, stretch=1)

    # -------------------------------------------------------------------------
    # TAB 3: Batch Experiment & Export
    # -------------------------------------------------------------------------
    def _build_tab_batch(self, parent: QWidget) -> None:
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top Action Bar
        top_bar = QHBoxLayout()
        self.btn_batch_select_files = QPushButton("Загрузить файлы / папку для пакета...")
        self.btn_batch_select_files.clicked.connect(self._on_batch_select_files_dialog)
        top_bar.addWidget(self.btn_batch_select_files)

        self.btn_batch_run_files = QPushButton("Пакетный расчёт выбранных файлов (0)")
        self.btn_batch_run_files.setEnabled(False)
        self.btn_batch_run_files.clicked.connect(self._on_batch_run_files)
        top_bar.addWidget(self.btn_batch_run_files)

        self.btn_batch_current_rois = QPushButton("Рассчитать все активные ROI")
        self.btn_batch_current_rois.clicked.connect(self._on_batch_calc_current_rois)
        top_bar.addWidget(self.btn_batch_current_rois)

        self.btn_clear_table = QPushButton("Очистить таблицу")
        self.btn_clear_table.clicked.connect(self._on_clear_batch_table)
        top_bar.addWidget(self.btn_clear_table)

        top_bar.addStretch()

        self.btn_export_csv = QPushButton("Экспорт в CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        top_bar.addWidget(self.btn_export_csv)

        self.btn_export_xlsx = QPushButton("Экспорт в Excel (.xlsx)")
        self.btn_export_xlsx.setStyleSheet("background-color: #1b633b; font-weight: 600; padding: 7px 14px;")
        self.btn_export_xlsx.clicked.connect(self._on_export_xlsx)
        top_bar.addWidget(self.btn_export_xlsx)

        self.btn_export_figs = QPushButton("Экспорт графиков (300 DPI PNG, SVG)")
        self.btn_export_figs.setStyleSheet("background-color: #5c3577; font-weight: 600; padding: 7px 14px;")
        self.btn_export_figs.clicked.connect(self._on_export_figures)
        top_bar.addWidget(self.btn_export_figs)

        main_layout.addLayout(top_bar)

        # Batch Table
        self.batch_table = QTableWidget(0, 12)
        self.batch_headers = [
            "Источник", "Участок (ROI)", "Метод", "τ", "d",
            "Max H1", "Mean H1", "Энтропия H0", "Энтропия H1",
            "H1 L2-норма", "Вассерштейн", "Боттлнек"
        ]
        self.batch_table.setHorizontalHeaderLabels(self.batch_headers)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.batch_table.setAlternatingRowColors(True)
        main_layout.addWidget(self.batch_table, stretch=1)

    # =========================================================================
    # Styling
    # =========================================================================

    def _apply_styling(self) -> None:
        """Apply modern dark scientific styling with clean typography."""
        qss = """
        QMainWindow, QWidget {
            background-color: #17181c;
            color: #d1d5db;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 12px;
        }
        QGroupBox {
            border: 1px solid #2e3342;
            border-radius: 6px;
            margin-top: 10px;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-weight: 600;
            font-size: 12px;
            color: #38bdf8;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #262933;
            border: 1px solid #3b4252;
            border-radius: 4px;
            padding: 6px 14px;
            color: #f1f5f9;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 12px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #323745;
            border-color: #38bdf8;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #1b1d24;
            border-color: #0284c7;
        }
        QPushButton:disabled {
            background-color: #1c1e24;
            color: #64748b;
            border-color: #2a2e38;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #21242c;
            border: 1px solid #373e4d;
            border-radius: 4px;
            padding: 5px 8px;
            color: #f8fafc;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 12px;
        }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
            border-color: #4f5b70;
        }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
            border-color: #38bdf8;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QTabWidget::pane {
            border: 1px solid #2e3342;
            background: #17181c;
            border-top: 1px solid #2e3342;
        }
        QTabBar::tab {
            background: #20232b;
            color: #94a3b8;
            padding: 8px 18px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            margin-right: 3px;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 12px;
            font-weight: 600;
        }
        QTabBar::tab:hover {
            background: #282c37;
            color: #e2e8f0;
        }
        QTabBar::tab:selected {
            background: #17181c;
            color: #38bdf8;
            border: 1px solid #2e3342;
            border-bottom: 2px solid #17181c;
        }
        QTableWidget {
            background-color: #1c1e24;
            gridline-color: #2b303d;
            border: 1px solid #2e3342;
            color: #e2e8f0;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 12px;
        }
        QHeaderView::section {
            background-color: #232731;
            color: #38bdf8;
            padding: 5px 8px;
            border: 1px solid #2e3342;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
        }
        QScrollBar:vertical {
            background: #1a1c22;
            width: 10px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #333a48;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background: #475266;
        }
        QStatusBar {
            background-color: #121317;
            color: #94a3b8;
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            font-size: 11px;
            border-top: 1px solid #232731;
        }
        QCheckBox {
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
            color: #cbd5e1;
            spacing: 6px;
        }
        QLabel {
            font-family: 'Segoe UI', 'Inter', 'IBM Plex Sans', sans-serif;
        }
        """
        self.setStyleSheet(qss)

    # =========================================================================
    # Data Loading & ROI Handlers
    # =========================================================================

    def _populate_presets_combo(self) -> None:
        """Populate preset combobox with synthetic demos, sample scans, and user presets."""
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItem("Выберите предустановку...", None)

        demos = get_demo_presets()
        for d in demos:
            self.combo_presets.addItem(d["label"], d)

        samples = discover_sample_scans()
        for s in samples:
            self.combo_presets.addItem(s["label"], s)

        user_presets = load_user_presets()
        for u in user_presets:
            lbl = u.get("label", "Пользовательский")
            self.combo_presets.addItem(f"Пресет: {lbl}", u)

        self.combo_presets.blockSignals(False)

    def _on_save_preset_dialog(self) -> None:
        """Save current ROI configuration as a user preset in user_presets.json."""
        if self.current_image_array is None or not self.rois:
            QMessageBox.information(self, "Сохранить пресет", "Нет активного скана или рамки ROI для сохранения.")
            return

        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Введите название нового пресета:")
        if not ok or not name.strip():
            return

        roi = self.rois[self.active_roi_index]
        pos = roi.pos()
        size = roi.size()
        h, w = self.current_image_array.shape

        preset_data = {
            "id": f"user_{name.strip()}",
            "label": name.strip(),
            "type": "user",
            "filename": self.current_image_path or "",
            "x_start_ratio": float(pos[0] / max(w, 1)),
            "x_end_ratio": float((pos[0] + size[0]) / max(w, 1)),
            "y_start_ratio": float(pos[1] / max(h, 1)),
            "y_end_ratio": float((pos[1] + size[1]) / max(h, 1)),
            "method": "viterbi" if self.combo_method.currentIndex() == 0 else "columnwise",
            "threshold": float(self.spin_thresh.value()),
            "smoothing": int(self.spin_smooth.value()),
            "tau": int(self.spin_tau.value()),
            "dim": int(self.spin_dim.value()),
        }
        save_user_preset(preset_data)
        self._populate_presets_combo()
        self.status_bar.showMessage(f"Пресет '{name.strip()}' успешно сохранён.")

    def _load_initial_sample(self) -> None:
        """Load first available sample scan or fallback to synthetic demo on startup."""
        samples = discover_sample_scans()
        if samples:
            self._load_ecg_by_config(samples[0])
            return

        demos = get_demo_presets()
        if demos:
            self._load_ecg_by_config(demos[0])

    def _on_preset_selected(self, index: int) -> None:
        if index <= 0:
            return
        cfg = self.combo_presets.itemData(index)
        if cfg:
            self._load_ecg_by_config(cfg)

    def _load_ecg_by_config(self, cfg: dict) -> None:
        cfg_type = cfg.get("type", "file")
        filename = cfg.get("filename", "")

        # 1. Synthetic Demo Presets (work on any machine with zero external files)
        if cfg_type == "synthetic" or str(filename).startswith("synthetic:"):
            ecg_type = cfg.get("ecg_type", "normal_sinus")
            img_rgb, _, _ = generate_synthetic_ecg(
                ecg_type=ecg_type, duration_s=4.0, fs=250, noise_level=0.02, add_grid=True
            )
            gray = np.mean(img_rgb, axis=2).astype(np.uint8)
            self.current_image_path = cfg.get("label", "Синтетический скан")
            self.current_image_array = gray
            self.image_item.setImage(gray)
            self.image_plot_widget.autoRange()

            self._clear_rois()
            h, w = gray.shape
            x0 = int(w * cfg.get("x_start_ratio", 0.05))
            x1 = int(w * cfg.get("x_end_ratio", 0.95))
            y0 = int(h * cfg.get("y_start_ratio", 0.15))
            y1 = int(h * cfg.get("y_end_ratio", 0.85))

            self._create_roi(pos=(x0, y0), size=(x1 - x0, y1 - y0), name=cfg.get("label", "Демо"))
            self.status_bar.showMessage(f"Загружен пресет: {cfg.get('label')}")
            return

        # 2. File-based Scans / User Presets
        filepath = cfg.get("filename") or cfg.get("image_path", "")
        if not filepath:
            return

        if not os.path.isfile(filepath):
            resolved = resource_path(filepath)
            if os.path.isfile(resolved):
                filepath = resolved
            else:
                self.status_bar.showMessage(f"Файл не найден: {filepath}. Загрузите скан через 'Открыть скан...'")
                return

        try:
            img = load_ecg_image(filepath)
            self.current_image_path = filepath
            self.current_image_array = img
            self.image_item.setImage(img)
            self.image_plot_widget.autoRange()

            self._clear_rois()
            h, w = img.shape
            x0 = int(w * cfg.get("x_start_ratio", 0.05))
            x1 = int(w * cfg.get("x_end_ratio", 0.95))
            y0 = int(h * cfg.get("y_start_ratio", 0.20))
            y1 = int(h * cfg.get("y_end_ratio", 0.50))

            if "threshold" in cfg:
                self.spin_thresh.setValue(cfg["threshold"])
            if "smoothing" in cfg:
                self.spin_smooth.setValue(cfg["smoothing"])
            if "tau" in cfg:
                self.spin_tau.setValue(cfg["tau"])
            if "dim" in cfg:
                self.spin_dim.setValue(cfg["dim"])

            self._create_roi(pos=(x0, y0), size=(x1 - x0, y1 - y0), name=cfg.get("label", "Участок"))
            self.status_bar.showMessage(f"Загружен скан: {os.path.basename(filepath)} ({cfg.get('label')})")
        except Exception as exc:
            self.status_bar.showMessage(f"Ошибка загрузки {filepath}: {exc}")

    def _on_open_scan_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите скан ЭКГ", "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if filepath:
            try:
                img = load_ecg_image(filepath)
                self.current_image_path = filepath
                self.current_image_array = img
                self.image_item.setImage(img)
                self.image_plot_widget.autoRange()

                self._clear_rois()
                h, w = img.shape
                # Default ROI in the middle
                self._create_roi(pos=(int(w * 0.1), int(h * 0.2)), size=(int(w * 0.8), int(h * 0.3)), name="Участок 1")
                self.status_bar.showMessage(f"Загружен файл: {os.path.basename(filepath)}")
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл:\n{exc}")

    def _on_mitbih_dialog(self) -> None:
        """Dialog to select MIT-BIH record from PhysioNet."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Загрузка записи PhysioNet MIT-BIH")
        layout = QFormLayout(dialog)

        combo_records = QComboBox()
        combo_records.addItems(list_mitbih_records())
        layout.addRow("Номер записи:", combo_records)

        spin_duration = QSpinBox()
        spin_duration.setRange(2, 60)
        spin_duration.setValue(10)
        layout.addRow("Длительность (с):", spin_duration)

        btn_load = QPushButton("Загрузить")
        layout.addRow(btn_load)

        def do_load():
            rec_id = combo_records.currentText()
            dur = spin_duration.value()
            dialog.accept()
            self._load_mitbih_record_async(rec_id, dur)

        btn_load.clicked.connect(do_load)
        dialog.exec()

    def _load_mitbih_record_async(self, record_name: str, duration_s: int) -> None:
        self.status_bar.showMessage(f"Загрузка MIT-BIH {record_name} из PhysioNet...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        class LoadThread(QThread):
            success = Signal(object, float)
            error = Signal(str)

            def run(self):
                try:
                    sig, fs = load_mitbih_record(record_name)
                    self.success.emit(sig, fs)
                except Exception as e:
                    self.error.emit(str(e))

        thread = LoadThread(self)

        def on_success(full_sig, fs):
            self.progress_bar.setVisible(False)
            max_pts = int(duration_s * fs)
            signal_slice = full_sig[:max_pts]
            signal_slice = (signal_slice - np.mean(signal_slice)) / (np.std(signal_slice) + 1e-8)

            self.current_signal = signal_slice
            self.current_image_path = f"MIT-BIH {record_name}"
            self._update_signal_plot(signal_slice, fs=fs)
            self._trigger_reembedding()
            self.status_bar.showMessage(f"MIT-BIH {record_name} загружен ({duration_s} с, {len(signal_slice)} отсчётов)")

        def on_error(err):
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Ошибка загрузки MIT-BIH", f"Не удалось загрузить запись:\n{err}")

        thread.success.connect(on_success)
        thread.error.connect(on_error)
        thread.start()

    def _on_synth_dialog(self) -> None:
        """Dialog to generate synthetic ECG scan."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Генератор синтетических сканов ЭКГ")
        layout = QFormLayout(dialog)

        combo_type = QComboBox()
        combo_type.addItem("Синусовый ритм (Норма)", "normal_sinus")
        combo_type.addItem("Желудочковая экстрасистолия", "extrasystole")
        combo_type.addItem("Фибрилляция предсердий", "afib")
        layout.addRow("Тип ритма:", combo_type)

        spin_dur = QDoubleSpinBox()
        spin_dur.setRange(2.0, 15.0)
        spin_dur.setValue(5.0)
        layout.addRow("Длительность (с):", spin_dur)

        spin_noise = QDoubleSpinBox()
        spin_noise.setRange(0.0, 0.2)
        spin_noise.setSingleStep(0.01)
        spin_noise.setValue(0.02)
        layout.addRow("Шум:", spin_noise)

        chk_grid = QCheckBox("Сетка миллиметровки")
        chk_grid.setChecked(True)
        layout.addRow(chk_grid)

        btn_gen = QPushButton("Сгенерировать скан")
        layout.addRow(btn_gen)

        def do_gen():
            ecg_type = combo_type.currentData()
            dur = spin_dur.value()
            noise = spin_noise.value()
            grid = chk_grid.isChecked()
            dialog.accept()

            img, gt, t = generate_synthetic_ecg(
                ecg_type=ecg_type, duration_s=dur, fs=500, noise_level=noise, add_grid=grid
            )
            self.current_image_path = f"Синтез ({combo_type.currentText()})"
            # Convert RGB image to grayscale for display & digitizer
            gray = np.mean(img, axis=2).astype(np.uint8)
            self.current_image_array = gray
            self.image_item.setImage(gray)
            self.image_plot_widget.autoRange()

            self._clear_rois()
            h, w = gray.shape
            self._create_roi(pos=(int(w * 0.05), int(h * 0.1)), size=(int(w * 0.9), int(h * 0.8)), name=combo_type.currentText())
            self.status_bar.showMessage(f"Сгенерирован синтетический скан: {combo_type.currentText()}")

        btn_gen.clicked.connect(do_gen)
        dialog.exec()

    # -------------------------------------------------------------------------
    # ROI Management
    # -------------------------------------------------------------------------

    def _create_roi(self, pos: Tuple[int, int], size: Tuple[int, int], name: str = "ROI") -> pg.RectROI:
        color_hex = self.roi_colors[len(self.rois) % len(self.roi_colors)]
        roi = pg.RectROI(pos, size, pen=pg.mkPen(color_hex, width=2.5))
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.sigRegionChangeFinished.connect(lambda: self._on_roi_changed(roi))
        self.image_plot_widget.addItem(roi)
        self.rois.append(roi)
        self.active_roi_index = len(self.rois) - 1
        self._on_roi_changed(roi)
        return roi

    def _on_add_roi(self) -> None:
        if self.current_image_array is None:
            return
        h, w = self.current_image_array.shape
        offset = len(self.rois) * 30
        self._create_roi(
            pos=(int(w * 0.2) + offset, int(h * 0.2) + offset),
            size=(int(w * 0.4), int(h * 0.3)),
            name=f"Участок {len(self.rois) + 1}",
        )

    def _on_delete_active_roi(self) -> None:
        if not self.rois:
            return
        roi = self.rois[self.active_roi_index]
        self.image_plot_widget.removeItem(roi)
        self.rois.pop(self.active_roi_index)
        self.active_roi_index = max(0, len(self.rois) - 1)
        if self.rois:
            self._on_roi_changed(self.rois[self.active_roi_index])

    def _clear_rois(self) -> None:
        for roi in self.rois:
            self.image_plot_widget.removeItem(roi)
        self.rois.clear()
        self.active_roi_index = 0

    def _on_roi_changed(self, roi: pg.RectROI) -> None:
        if roi in self.rois:
            self.active_roi_index = self.rois.index(roi)
        self._trigger_reanalysis()

    # -------------------------------------------------------------------------
    # Processing Pipeline (Digitization -> Embedding -> TDA)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Processing Pipeline (Digitization -> Embedding -> TDA)
    # -------------------------------------------------------------------------

    def _trigger_reanalysis(self) -> None:
        """Digitize the active ROI and proceed to embedding."""
        if self.current_image_array is None or not self.rois:
            return

        roi = self.rois[self.active_roi_index]
        pos = roi.pos()
        size = roi.size()

        h, w = self.current_image_array.shape
        x0 = max(0, int(pos[0]))
        y0 = max(0, int(pos[1]))
        x1 = min(w, int(pos[0] + size[0]))
        y1 = min(h, int(pos[1] + size[1]))

        if (x1 - x0) < 20 or (y1 - y0) < 10:
            return

        method = "viterbi" if self.combo_method.currentIndex() == 0 else "columnwise"
        thresh = float(self.spin_thresh.value())
        smooth = self.spin_smooth.value()
        if smooth % 2 == 0:
            smooth += 1

        self.status_bar.showMessage("Оцифровка участка...")
        self.progress_bar.setVisible(True)

        worker = DigitizationWorker(
            image_or_path=self.current_image_array,
            y_start=y0,
            y_end=y1,
            x_start=x0,
            x_end=x1,
            method=method,
            brightness_threshold=thresh,
            smoothing_window=smooth,
            continuity_weight=0.01,
        )

        worker.signals.finished.connect(self._on_digitization_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_digitization_finished(self, signal: np.ndarray) -> None:
        self.progress_bar.setVisible(False)
        self.current_signal = signal
        self._update_signal_plot(signal)
        self._trigger_reembedding()

    def _update_signal_plot(self, signal: np.ndarray, fs: Optional[float] = None) -> None:
        if fs is None:
            # Derive time in ms from calibration
            time_ms = np.array([self.calibration.px_to_ms(i) for i in range(len(signal))])
            self.signal_curve.setData(time_ms, signal)
            self.signal_plot_widget.setLabel("bottom", "Время (мс)")
        else:
            time_s = np.arange(len(signal)) / fs
            self.signal_curve.setData(time_s, signal)
            self.signal_plot_widget.setLabel("bottom", "Время (с)")

    def _trigger_reembedding(self) -> None:
        """Run Takens embedding and Ripser persistence on current signal."""
        if self.current_signal is None or len(self.current_signal) < 20:
            return

        tau = self.spin_tau.value()
        dim = self.spin_dim.value()

        self.status_bar.showMessage(f"Расчёт TDA: Takens (τ={tau}, d={dim}) + Ripser...")
        self.progress_bar.setVisible(True)

        worker = AnalysisWorker(
            signal=self.current_signal,
            delay=tau,
            dimension=dim,
            reference_dgms=self.reference_dgms,
        )
        worker.signals.finished.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_analysis_finished(self, result: dict) -> None:
        self.progress_bar.setVisible(False)
        self.current_cloud = result["cloud"]
        self.current_dgms = result["dgms"]
        self.current_metrics = result["metrics"]

        # Update 3D Attractor
        self._update_3d_attractor(self.current_cloud)

        # Update Persistence Diagram
        self._update_persistence_diagram(self.current_dgms)

        # Update Live Metrics Table
        self._update_metrics_table(self.current_metrics)

        # Update Tab 2 parameter estimation curves
        self._update_parameter_curves()

        self.status_bar.showMessage("Анализ завершён успешно.")

    def _on_worker_error(self, err_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Ошибка вычислений")
        print("Worker error:\n", err_msg, file=sys.stderr)

    # -------------------------------------------------------------------------
    # Visualization Updates
    # -------------------------------------------------------------------------

    def _update_3d_attractor(self, cloud: np.ndarray) -> None:
        if not self.has_opengl:
            # 2D fallback
            self.gl_fallback_plot.clear()
            self.gl_fallback_plot.plot(cloud[:, 0], cloud[:, 1], pen=pg.mkPen("#3388ff", width=1))
            return

        n_pts = len(cloud)
        if n_pts == 0:
            return

        # Prepare 3D coordinates (pad with 0 if d=2)
        if cloud.shape[1] >= 3:
            pts = cloud[:, :3]
        else:
            pts = np.column_stack([cloud[:, 0], cloud[:, 1], np.zeros(n_pts)])

        # Time gradient colors: Viridis/plasma-like transition
        t_norm = np.linspace(0, 1, n_pts)
        colors = np.zeros((n_pts, 4))
        # Blue -> Cyan -> Yellow -> Red gradient
        colors[:, 0] = t_norm
        colors[:, 1] = np.sin(np.pi * t_norm)
        colors[:, 2] = 1.0 - t_norm
        colors[:, 3] = 0.8  # alpha

        self.gl_scatter.setData(pos=pts, color=colors, size=4.0, pxMode=True)

    def _rotate_3d_view(self) -> None:
        if self.has_opengl:
            self.current_azimuth = (self.current_azimuth + 1.0) % 360.0
            self.gl_view.setCameraPosition(azimuth=self.current_azimuth)

    def _toggle_rotation(self, checked: bool) -> None:
        if checked:
            self.rotation_timer.start(50)
        else:
            self.rotation_timer.stop()

    def _update_persistence_diagram(self, dgms: List[np.ndarray]) -> None:
        h0 = dgms[0]
        h1 = dgms[1]

        # Filter finite
        h0_fin = h0[np.isfinite(h0).all(axis=1)] if len(h0) > 0 else np.empty((0, 2))
        h1_fin = h1[np.isfinite(h1).all(axis=1)] if len(h1) > 0 else np.empty((0, 2))

        # Max range for diagonal
        max_val = 1.0
        if len(h0_fin) > 0:
            max_val = max(max_val, float(np.max(h0_fin)))
        if len(h1_fin) > 0:
            max_val = max(max_val, float(np.max(h1_fin)))

        self.dgm_diagonal.setData([0, max_val * 1.1], [0, max_val * 1.1])
        self.dgm_h0_scatter.setData(pos=h0_fin)
        self.dgm_h1_scatter.setData(pos=h1_fin)
        self.dgm_plot_widget.setXRange(0, max_val * 1.1)
        self.dgm_plot_widget.setYRange(0, max_val * 1.1)

    def _update_metrics_table(self, metrics: Dict[str, float]) -> None:
        keys = [
            "max_h1_persistence", "mean_h1_persistence", "persistent_entropy_h0",
            "persistent_entropy_h1", "h1_l2_norm", "wasserstein_distance", "bottleneck_distance"
        ]
        for col, k in enumerate(keys):
            val = metrics.get(k, float("nan"))
            txt = f"{val:.4f}" if np.isfinite(val) else "—"
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignCenter)
            self.table_metrics.setItem(0, col, item)

    def _on_calibration_changed(self) -> None:
        px_per_mm = self.spin_px_per_mm.value()
        speed = 25.0 if self.combo_speed.currentIndex() == 0 else 50.0
        self.calibration = CalibrationParams(px_per_mm=px_per_mm, paper_speed_mm_per_s=speed)
        if self.current_signal is not None:
            self._update_signal_plot(self.current_signal)

    # -------------------------------------------------------------------------
    # Reference Standard (Norm) Handling
    # -------------------------------------------------------------------------

    def _on_set_reference(self) -> None:
        if self.current_dgms is None:
            QMessageBox.information(self, "Инфо", "Сначала дождитесь расчёта персистентности для текущего участка.")
            return

        self.reference_dgms = self.current_dgms
        src_name = os.path.basename(self.current_image_path or "Скан")
        self.reference_label = f"{src_name} (ROI #{self.active_roi_index + 1})"
        self.lbl_reference_status.setText(f"Эталон: {self.reference_label}")
        self.lbl_reference_status.setStyleSheet("color: #33cc33; font-weight: bold;")
        self.status_bar.showMessage(f"Зафиксирован эталон: {self.reference_label}")

        # Re-run embedding to calculate distances against new reference
        self._trigger_reembedding()

    # -------------------------------------------------------------------------
    # Tab 2: Parameter Estimation & AAFT Test
    # -------------------------------------------------------------------------

    def _on_auto_tau_click(self) -> None:
        if self.current_signal is None:
            return
        ami_vals, optimal_tau = compute_ami(self.current_signal, max_lag=40)
        self.spin_tau.setValue(optimal_tau)
        self.status_bar.showMessage(f"Оптимальный τ по 1-му минимуму AMI: {optimal_tau}")

    def _update_parameter_curves(self) -> None:
        if self.current_signal is None or len(self.current_signal) < 40:
            return

        worker = ParameterEstimationWorker(self.current_signal, max_lag=40, max_dim=6)

        def on_param_done(res: dict):
            # AMI
            ami_vals = res["ami_vals"]
            opt_tau = res["optimal_tau"]
            lags = np.arange(1, len(ami_vals) + 1)
            self.ami_curve.setData(lags, ami_vals)
            self.ami_marker.setValue(opt_tau)
            self.lbl_ami_opt.setText(f"Оптимальный τ (1-й локальный минимум AMI): {opt_tau}")

            # FNN
            fnn_vals = res["fnn_vals"] * 100.0  # in percent
            opt_d = res["optimal_d"]
            dims = np.arange(1, len(fnn_vals) + 1)
            self.fnn_curve.setData(dims, fnn_vals)
            self.lbl_fnn_opt.setText(f"Оптимальная размерность вложения d (FNN < 1%): {opt_d}")

        worker.signals.finished.connect(on_param_done)
        self.thread_pool.start(worker)

    def _on_run_surrogate_test(self) -> None:
        if self.current_signal is None:
            QMessageBox.information(self, "Инфо", "Сначала выделите сигнал на вкладке 1.")
            return

        n_surr = self.spin_surr_n.value()
        metric_name = self.combo_surr_metric.currentText()
        tau = self.spin_tau.value()
        dim = self.spin_dim.value()

        self.status_bar.showMessage(f"Генерация {n_surr} AAFT суррогатов и расчёт теста значимости...")
        self.btn_run_surr.setEnabled(False)
        self.progress_bar.setVisible(True)

        worker = SurrogateWorker(
            signal=self.current_signal,
            delay=tau,
            dimension=dim,
            n_surrogates=n_surr,
            metric_name=metric_name,
        )

        def on_surr_done(res: dict):
            self.btn_run_surr.setEnabled(True)
            self.progress_bar.setVisible(False)

            real_val = res["real_value"]
            surrs = np.array(res["surrogate_values"])
            z_score = res["z_score"]
            p_val = res["p_value"]

            # Histogram plot
            self.surr_plot.clear()
            self.surr_real_line.setValue(real_val)
            self.surr_plot.addItem(self.surr_real_line)

            y, x = np.histogram(surrs, bins=10)
            bargraph = pg.BarGraphItem(
                x0=x[:-1], x1=x[1:], height=y, brush=pg.mkBrush("#3388ff", alpha=180), pen=pg.mkPen("#ffffff", width=1)
            )
            self.surr_plot.addItem(bargraph)

            # Update text labels
            self.lbl_surr_real.setText(f"Реальный сигнал ({metric_name}): {real_val:.4f}")
            self.lbl_surr_mean.setText(f"Суррогаты (Mean ± Std): {surrs.mean():.4f} ± {surrs.std():.4f}")
            self.lbl_surr_z.setText(f"Z-оценка: {z_score:.2f}")
            self.lbl_surr_p.setText(f"p-value: {p_val:.4e}")

            if p_val < 0.05:
                self.lbl_surr_verdict.setText("Заключение: динамика нелинейна, доминирующий цикл статистически значим (p < 0.05)")
                self.lbl_surr_verdict.setStyleSheet("font-size: 13px; font-weight: 600; color: #4ade80;")
            else:
                self.lbl_surr_verdict.setText("Заключение: гипотеза линейного шума не отвергается (p >= 0.05)")
                self.lbl_surr_verdict.setStyleSheet("font-size: 13px; font-weight: 600; color: #fbbf24;")

            self.status_bar.showMessage("Тест суррогатных данных завершён.")

        def on_surr_error(err):
            self.btn_run_surr.setEnabled(True)
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Ошибка теста суррогатов", err)

        worker.signals.finished.connect(on_surr_done)
        worker.signals.error.connect(on_surr_error)
        self.thread_pool.start(worker)

    # -------------------------------------------------------------------------
    # Tab 3: Batch Experiment & Export
    # -------------------------------------------------------------------------

    def _on_add_to_batch_table(self) -> None:
        if not self.current_metrics:
            QMessageBox.information(self, "Инфо", "Нет данных для добавления.")
            return

        src = os.path.basename(self.current_image_path or "Неизвестно")
        roi_name = f"ROI #{self.active_roi_index + 1}"
        method = "Витерби" if self.combo_method.currentIndex() == 0 else "Поколоночный"
        tau = self.spin_tau.value()
        dim = self.spin_dim.value()

        row_dict = {
            "Источник": src,
            "Участок (ROI)": roi_name,
            "Метод": method,
            "τ": tau,
            "d": dim,
            "Max H1": self.current_metrics.get("max_h1_persistence", np.nan),
            "Mean H1": self.current_metrics.get("mean_h1_persistence", np.nan),
            "Энтропия H0": self.current_metrics.get("persistent_entropy_h0", np.nan),
            "Энтропия H1": self.current_metrics.get("persistent_entropy_h1", np.nan),
            "H1 L2-норма": self.current_metrics.get("h1_l2_norm", np.nan),
            "Вассерштейн": self.current_metrics.get("wasserstein_distance", np.nan),
            "Боттлнек": self.current_metrics.get("bottleneck_distance", np.nan),
        }

        self.batch_dataset_rows.append(row_dict)
        self._insert_row_into_batch_table(row_dict)
        self.status_bar.showMessage(f"Добавлено в таблицу: {src} {roi_name}")

    def _insert_row_into_batch_table(self, row_dict: Dict[str, Any]) -> None:
        row_idx = self.batch_table.rowCount()
        self.batch_table.insertRow(row_idx)

        for col_idx, key in enumerate(self.batch_headers):
            val = row_dict.get(key, "")
            if isinstance(val, float):
                txt = f"{val:.4f}" if np.isfinite(val) else "—"
            else:
                txt = str(val)
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignCenter)
            self.batch_table.setItem(row_idx, col_idx, item)

    def _on_clear_batch_table(self) -> None:
        self.batch_table.setRowCount(0)
        self.batch_dataset_rows.clear()
        self.status_bar.showMessage("Таблица пакетного эксперимента очищена.")

    def _on_batch_select_files_dialog(self) -> None:
        """Select multiple ECG image files or a folder for batch analysis."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите сканы ЭКГ для пакетного анализа",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.batch_selected_files = files
            count = len(files)
            self.btn_batch_run_files.setText(f"Пакетный расчёт выбранных файлов ({count})")
            self.btn_batch_run_files.setEnabled(True)
            self.status_bar.showMessage(f"Выбрано {count} файлов для пакетного анализа.")

    def _on_batch_run_files(self) -> None:
        """Run batch processing across all user-selected image files."""
        if not self.batch_selected_files:
            return

        files = list(self.batch_selected_files)
        self.status_bar.showMessage(f"Пакетный расчёт {len(files)} файлов...")
        self.progress_bar.setVisible(True)

        tau = self.spin_tau.value()
        dim = self.spin_dim.value()
        method = "viterbi" if self.combo_method.currentIndex() == 0 else "columnwise"

        class BatchFilesThread(QThread):
            progress = Signal(dict)
            done = Signal()

            def run(self):
                ref_dgm = None
                for i, filepath in enumerate(files):
                    try:
                        img = load_ecg_image(filepath)
                        h, w = img.shape
                        x0, x1 = int(w * 0.05), int(w * 0.95)
                        y0, y1 = int(h * 0.20), int(h * 0.50)

                        if method == "viterbi":
                            from core.digitizer import extract_signal_viterbi
                            sig = extract_signal_viterbi(
                                img, y_start=y0, y_end=y1, x_start=x0, x_end=x1, continuity_weight=0.01
                            )
                        else:
                            from core.digitizer import extract_signal_columnwise
                            sig = extract_signal_columnwise(img, y_start=y0, y_end=y1, x_start=x0, x_end=x1)

                        cloud = takens_embedding(sig, delay=tau, dimension=dim)
                        res = compute_persistence(cloud, maxdim=1)
                        dgms = res["dgms"]

                        if i == 0 and ref_dgm is None:
                            ref_dgm = dgms

                        metrics_dict = {}
                        for m_name in get_all_metrics():
                            try:
                                metrics_dict[m_name] = compute_metric(
                                    m_name, dgms=dgms, reference_dgms=ref_dgm, dim=1
                                )
                            except Exception:
                                metrics_dict[m_name] = np.nan

                        row = {
                            "Источник": os.path.basename(filepath),
                            "Участок (ROI)": "Центральная полоса",
                            "Метод": "Витерби" if method == "viterbi" else "Поколоночный",
                            "τ": tau,
                            "d": dim,
                            "Max H1": metrics_dict.get("max_h1_persistence", np.nan),
                            "Mean H1": metrics_dict.get("mean_h1_persistence", np.nan),
                            "Энтропия H0": metrics_dict.get("persistent_entropy_h0", np.nan),
                            "Энтропия H1": metrics_dict.get("persistent_entropy_h1", np.nan),
                            "H1 L2-норма": metrics_dict.get("h1_l2_norm", np.nan),
                            "Вассерштейн": metrics_dict.get("wasserstein_distance", np.nan),
                            "Боттлнек": metrics_dict.get("bottleneck_distance", np.nan),
                        }
                        self.progress.emit(row)
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}", file=sys.stderr)
                self.done.emit()

        thread = BatchFilesThread(self)

        def on_prog(row):
            self.batch_dataset_rows.append(row)
            self._insert_row_into_batch_table(row)

        def on_done():
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Пакетный расчёт файлов завершён.")

        thread.progress.connect(on_prog)
        thread.done.connect(on_done)
        thread.start()

    def _on_batch_calc_current_rois(self) -> None:
        """Calculate TDA metrics for all active ROIs on current scan."""
        if self.current_image_array is None or not self.rois:
            QMessageBox.information(
                self,
                "Расчёт активных ROI",
                "На вкладке '1. Интерактивный анализ' нет загруженного скана или активных рамок (ROI).\n"
                "Добавьте хотя бы одну рамку на скан."
            )
            return

        self.status_bar.showMessage(f"Расчёт {len(self.rois)} активных ROI...")
        self.progress_bar.setVisible(True)

        tau = self.spin_tau.value()
        dim = self.spin_dim.value()
        method = "viterbi" if self.combo_method.currentIndex() == 0 else "columnwise"
        thresh = float(self.spin_thresh.value())
        smooth = int(self.spin_smooth.value())
        if smooth % 2 == 0:
            smooth += 1

        img = self.current_image_array
        h, w = img.shape
        source_name = os.path.basename(self.current_image_path) if self.current_image_path else "Текущий скан"

        roi_data_list = []
        for idx, r in enumerate(self.rois):
            pos = r.pos()
            sz = r.size()
            x0 = max(0, int(pos[0]))
            y0 = max(0, int(pos[1]))
            x1 = min(w, int(pos[0] + sz[0]))
            y1 = min(h, int(pos[1] + sz[1]))
            roi_data_list.append((idx + 1, x0, y0, x1, y1))

        class BatchROIsThread(QThread):
            progress = Signal(dict)
            done = Signal()

            def run(self):
                ref_dgm = None
                for num, x0, y0, x1, y1 in roi_data_list:
                    try:
                        if method == "viterbi":
                            from core.digitizer import extract_signal_viterbi
                            sig = extract_signal_viterbi(
                                img, y_start=y0, y_end=y1, x_start=x0, x_end=x1,
                                brightness_threshold=thresh, smoothing_window=smooth, continuity_weight=0.01
                            )
                        else:
                            from core.digitizer import extract_signal_columnwise
                            sig = extract_signal_columnwise(
                                img, y_start=y0, y_end=y1, x_start=x0, x_end=x1,
                                brightness_threshold=thresh, smoothing_window=smooth
                            )

                        cloud = takens_embedding(sig, delay=tau, dimension=dim)
                        res = compute_persistence(cloud, maxdim=1)
                        dgms = res["dgms"]

                        if ref_dgm is None:
                            ref_dgm = dgms

                        metrics_dict = {}
                        for m_name in get_all_metrics():
                            try:
                                metrics_dict[m_name] = compute_metric(
                                    m_name, dgms=dgms, reference_dgms=ref_dgm, dim=1
                                )
                            except Exception:
                                metrics_dict[m_name] = np.nan

                        row = {
                            "Источник": source_name,
                            "Участок (ROI)": f"ROI #{num} ({x0}:{x1}, {y0}:{y1})",
                            "Метод": "Витерби" if method == "viterbi" else "Поколоночный",
                            "τ": tau,
                            "d": dim,
                            "Max H1": metrics_dict.get("max_h1_persistence", np.nan),
                            "Mean H1": metrics_dict.get("mean_h1_persistence", np.nan),
                            "Энтропия H0": metrics_dict.get("persistent_entropy_h0", np.nan),
                            "Энтропия H1": metrics_dict.get("persistent_entropy_h1", np.nan),
                            "H1 L2-норма": metrics_dict.get("h1_l2_norm", np.nan),
                            "Вассерштейн": metrics_dict.get("wasserstein_distance", np.nan),
                            "Боттлнек": metrics_dict.get("bottleneck_distance", np.nan),
                        }
                        self.progress.emit(row)
                    except Exception as e:
                        print(f"Error processing ROI #{num}: {e}", file=sys.stderr)
                self.done.emit()

        thread = BatchROIsThread(self)

        def on_prog(row):
            self.batch_dataset_rows.append(row)
            self._insert_row_into_batch_table(row)

        def on_done():
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Расчёт активных ROI завершён.")

        thread.progress.connect(on_prog)
        thread.done.connect(on_done)
        thread.start()

    def _on_export_csv(self) -> None:
        if not self.batch_dataset_rows:
            QMessageBox.information(self, "Инфо", "Таблица пуста. Нечего экспортировать.")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "tda_ecg_metrics.csv", "CSV (*.csv)")
        if filepath:
            try:
                export_table_csv(self.batch_dataset_rows, filepath)
                QMessageBox.information(self, "Успех", f"Таблица успешно сохранена в:\n{filepath}")
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _on_export_xlsx(self) -> None:
        if not self.batch_dataset_rows:
            QMessageBox.information(self, "Инфо", "Таблица пуста. Нечего экспортировать.")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить Excel", "tda_ecg_metrics.xlsx", "Excel (*.xlsx)")
        if filepath:
            try:
                export_table_xlsx(self.batch_dataset_rows, filepath)
                QMessageBox.information(self, "Успех", f"Таблица успешно сохранена в:\n{filepath}")
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _on_export_figures(self) -> None:
        """Export publication-ready figures (PNG 300 DPI + SVG)."""
        if self.current_signal is None or self.current_cloud is None or self.current_dgms is None:
            QMessageBox.information(self, "Инфо", "Нет рассчитанных данных для экспорта фигур.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить фигуру для курсовой (без расширения)", "ecg_tda_figure", "Все файлы (*)"
        )
        if filepath:
            # strip .png or .svg if user typed it
            base_path = os.path.splitext(filepath)[0]
            try:
                tau = self.spin_tau.value()
                dim = self.spin_dim.value()
                paths = export_figure_publication(
                    signal=self.current_signal,
                    point_cloud=self.current_cloud,
                    dgms=self.current_dgms,
                    metrics=self.current_metrics,
                    filepath_base=base_path,
                    title="Топологический анализ ЭКГ (TopoCardio Lab)",
                    delay=tau,
                    dimension=dim,
                    dpi=300,
                )
                QMessageBox.information(
                    self, "Фигуры экспортированы",
                    f"Успешно сохранены публикации-готовые фигуры:\n• Растровый: {paths['png']} (300 DPI)\n• Векторный: {paths['svg']}"
                )
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def closeEvent(self, event) -> None:
        """Ensure all background workers are stopped on close."""
        self.rotation_timer.stop()
        self.thread_pool.waitForDone(500)
        super().closeEvent(event)
