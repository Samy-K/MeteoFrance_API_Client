"""
GUI entry point for MeteoFrance API Client.

Launches the PyQt6 application window.  All interactive CLI prompts are
replaced by the graphical interface defined in src/gui/.

Usage:
    python main_gui.py

Author:  Samy KRAIEM
Created: 2026
Updated: 2026
"""

import logging
import sys

# QtWebEngineWidgets MUST be imported before QApplication is instantiated.
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> None:
    """Create the QApplication and show the main window."""
    # Configure root logger before any module emits records
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)
    app.setApplicationName("MeteoFrance API Client")
    app.setApplicationVersion("2026")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
