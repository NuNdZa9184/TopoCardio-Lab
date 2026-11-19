"""
TopoCardio Lab — Топологический анализ данных (TDA) ЭКГ.
Главная точка входа приложения.
"""

from __future__ import annotations

import ctypes
import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication
from core.datasets import resource_path
from ui.main_window import MainWindow


def main() -> None:
    # Set Windows AppUserModelID so the taskbar uses the custom icon (Windows only)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "topocardio.lab.tda.1.0"
            )
        except Exception:
            pass

    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("TopoCardio Lab")
    app.setOrganizationName("Scientific TDA Lab")

    # Set application icon from TopoCardio_Lab_icon.svg
    icon_path = resource_path("TopoCardio_Lab_icon.svg")
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)

    # Set default clean typography (Segoe UI / Inter / IBM Plex Sans)
    app_font = QFont("Segoe UI", 10)
    app_font.setStyleHint(QFont.SansSerif)
    app.setFont(app_font)

    window = MainWindow()
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
