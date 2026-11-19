"""
TopoCardio Lab — Publication-grade Export Utilities.
Exports scientific figures (300 DPI PNG and vector SVG) with Russian typography
and summary tables (CSV / Excel).
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for export
import matplotlib.pyplot as plt
from persim import plot_diagrams
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side


def export_figure_publication(
    signal: np.ndarray,
    point_cloud: np.ndarray,
    dgms: List[np.ndarray],
    metrics: Dict[str, float],
    filepath_base: str,
    title: str = "Топологический анализ ЭКГ",
    delay: int = 6,
    dimension: int = 3,
    dpi: int = 300,
) -> Dict[str, str]:
    """Generate and save publication-ready figures in both .png and .svg format.

    Parameters
    ----------
    signal : np.ndarray
        Digitized 1-D ECG signal.
    point_cloud : np.ndarray
        Takens delay coordinate embedding.
    dgms : list[np.ndarray]
        Persistence diagrams [H0, H1].
    metrics : dict[str, float]
        Dictionary of calculated TDA metrics.
    filepath_base : str
        Path without extension (e.g. 'reports/ecg_analysis').
    title : str
        Figure title in Russian.
    delay : int
        Tau parameter.
    dimension : int
        Dimension d.
    dpi : int
        Resolution for raster export.

    Returns
    -------
    dict
        Paths to generated 'png' and 'svg' files.
    """
    # Configure clean publication typography
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle(title, fontweight="bold", fontsize=14)

    # 1. Signal plot
    ax1 = fig.add_subplot(1, 3, 1)
    time_pts = np.arange(len(signal))
    ax1.plot(time_pts, signal, color="#1f77b4", lw=1.2, label=r"$x(t)$")
    ax1.set_title("Оцифрованный сигнал ЭКГ", pad=8)
    ax1.set_xlabel("Индекс отсчёта $t$")
    ax1.set_ylabel("Нормированная амплитуда (Z-score)")
    ax1.legend(loc="upper right", framealpha=0.85)

    # 2. Takens attractor (2D projection with time colormap)
    ax2 = fig.add_subplot(1, 3, 2)
    t_idx = np.arange(len(point_cloud))
    ax2.plot(point_cloud[:, 0], point_cloud[:, 1], color="gray", alpha=0.35, lw=0.6)
    scatter = ax2.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        c=t_idx,
        cmap="viridis",
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.02, fraction=0.046)
    cbar.set_label("Время $t$", fontsize=9)
    ax2.set_title(f"Вложение Такенса ($\\tau={delay}$, $d={dimension}$)", pad=8)
    ax2.set_xlabel(r"$x(t)$")
    ax2.set_ylabel(rf"$x(t + {delay})$")

    # 3. Persistence diagram
    ax3 = fig.add_subplot(1, 3, 3)
    plot_diagrams(dgms, ax=ax3, show=False, legend=True)
    
    # Format metrics box text in Russian
    max_h1 = metrics.get("max_h1_persistence", 0.0)
    entropy_h1 = metrics.get("persistent_entropy_h1", 0.0)
    l2_norm = metrics.get("h1_l2_norm", 0.0)
    
    metrics_text = (
        f"Max $H_1$: {max_h1:.3f}\n"
        f"Энтропия $H_1$: {entropy_h1:.3f}\n"
        f"$L_2$-норма $H_1$: {l2_norm:.3f}"
    )
    ax3.text(
        0.05, 0.95, metrics_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc")
    )
    ax3.set_title("Диаграмма персистентности ($H_0, H_1$)", pad=8)
    ax3.set_xlabel("Время рождения (Birth)")
    ax3.set_ylabel("Время смерти (Death)")

    os.makedirs(os.path.dirname(os.path.abspath(filepath_base)) or ".", exist_ok=True)
    png_path = f"{filepath_base}.png"
    svg_path = f"{filepath_base}.svg"

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    return {"png": png_path, "svg": svg_path}


def export_table_csv(data_rows: List[Dict[str, Any]], filepath: str) -> str:
    """Export summary metrics table to CSV file."""
    if not data_rows:
        return filepath

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    fieldnames = list(data_rows[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in data_rows:
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.4f}"
                else:
                    formatted_row[k] = str(v)
            writer.writerow(formatted_row)

    return filepath


def export_table_xlsx(data_rows: List[Dict[str, Any]], filepath: str) -> str:
    """Export summary metrics table to styled Excel (.xlsx) file."""
    if not data_rows:
        return filepath

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TDA Метрики ЭКГ"

    headers = list(data_rows[0].keys())
    ws.append(headers)

    # Style header row
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    thin_border = Border(
        left=Side(style="thin", color="D3D3D3"),
        right=Side(style="thin", color="D3D3D3"),
        top=Side(style="thin", color="D3D3D3"),
        bottom=Side(style="thin", color="D3D3D3")
    )

    for col_idx, col in enumerate(ws.iter_cols(min_row=1, max_row=1, min_col=1, max_col=len(headers)), start=1):
        for cell in col:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border

    # Add rows
    for row_idx, row in enumerate(data_rows, start=2):
        row_values = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                row_values.append(round(v, 4))
            else:
                row_values.append(v)
        ws.append(row_values)

        fill_color = "F2F5F9" if row_idx % 2 == 0 else "FFFFFF"
        row_fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.fill = row_fill
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=10)
            if isinstance(cell.value, (int, float)):
                cell.alignment = Alignment(horizontal="right")
            else:
                cell.alignment = Alignment(horizontal="left")

    # Auto-adjust column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    wb.save(filepath)
    return filepath
