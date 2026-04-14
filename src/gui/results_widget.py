"""
Results panel shown after a successful data download.

Displays a station information summary, a scrollable preview of the
first rows of the processed dataset, and buttons to open the generated
PDF reports in the system default viewer.

Author:  Samy KRAIEM
Created: 2026
Updated: 2026
"""

import logging
import os
import subprocess
import sys

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data_handler import DatasetManager

logger = logging.getLogger(__name__)

_PREVIEW_ROWS = 50   # number of dataset rows shown in the table


class ResultsWidget(QWidget):
    """Panel that summarises a completed acquisition run.

    Call :meth:`populate` after a successful download to fill the widget.
    The widget is hidden by default and should be shown once data arrives.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_dir = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # --- Station info summary ----------------------------------------
        info_box = QGroupBox("Station information")
        self._info_lay = QVBoxLayout(info_box)
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._info_lay.addWidget(self._info_label)
        root.addWidget(info_box)

        # --- Dataset preview ---------------------------------------------
        preview_box = QGroupBox(f"Dataset preview (first {_PREVIEW_ROWS} rows)")
        preview_lay = QVBoxLayout(preview_box)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_lay.addWidget(self._table)
        root.addWidget(preview_box)

        # --- Open-report buttons -----------------------------------------
        btn_box = QGroupBox("Reports")
        btn_lay = QHBoxLayout(btn_box)

        self._btn_qc       = QPushButton("Open quality_checks.pdf")
        self._btn_factsheet = QPushButton("Open factsheet.pdf")
        self._btn_folder    = QPushButton("Open output folder")

        self._btn_qc.clicked.connect(lambda: self._open_pdf("quality_checks.pdf"))
        self._btn_factsheet.clicked.connect(lambda: self._open_pdf("factsheet.pdf"))
        self._btn_folder.clicked.connect(self._open_folder)

        for btn in (self._btn_qc, self._btn_factsheet, self._btn_folder):
            btn_lay.addWidget(btn)

        root.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(
        self,
        dataset: DatasetManager,
        station_info: pd.DataFrame,
        start_year: int,
        end_year: int,
        output_dir: str,
        source_label: str = '',
    ) -> None:
        """Fill the widget with the results of a completed run.

        Args:
            dataset (DatasetManager): Processed hourly dataset.
            station_info (pd.DataFrame): One-row station metadata DataFrame.
            start_year (int): First year of the downloaded period.
            end_year (int): Last year of the downloaded period.
            output_dir (str): Directory where CSVs and PDFs were written.
            source_label (str): Human-readable source name (e.g.
                ``'Météo-France (DPClim)'``).
        """
        self._output_dir = output_dir

        row = station_info.iloc[0]

        lat_raw = row.get('Latitude')
        lon_raw = row.get('Longitude')
        alt_raw = row.get('Altitude')
        lat = f"{float(lat_raw):.4f}" if pd.notna(lat_raw) else '—'
        lon = f"{float(lon_raw):.4f}" if pd.notna(lon_raw) else '—'
        alt = f"{float(alt_raw):.0f} m" if pd.notna(alt_raw) else '—'

        date_debut_raw = row.get('DateDebut', '')
        date_fin_raw   = row.get('DateFin',   '')
        date_debut = str(date_debut_raw)[:10] if pd.notna(date_debut_raw) else '—'
        date_fin   = str(date_fin_raw)[:10]   if pd.notna(date_fin_raw)   else 'present'

        sep = '&nbsp;&nbsp;|&nbsp;&nbsp;'
        src_part = f"{sep}<b>Source:</b> {source_label}" if source_label else ''
        self._info_label.setText(
            f"<b>ID:</b> {row.get('ID', '—')}"
            f"{sep}<b>Name:</b> {row.get('Nom', '—')}"
            f"{sep}<b>Alt:</b> {alt}"
            f"<br/>"
            f"<b>Lat:</b> {lat}"
            f"{sep}<b>Lon:</b> {lon}"
            f"{sep}<b>Station period:</b> {date_debut} – {date_fin}"
            f"<br/>"
            f"<b>Downloaded:</b> {start_year} – {end_year}"
            f"{src_part}"
            f"<br/>"
            f"<b>Output:</b> {os.path.abspath(output_dir)}"
        )

        # Fill dataset preview table
        df = dataset.data.head(_PREVIEW_ROWS)
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels(list(df.columns))
        self._table.setRowCount(len(df))

        for r, (_, data_row) in enumerate(df.iterrows()):
            for c, col in enumerate(df.columns):
                val  = data_row[col]
                if pd.isna(val):
                    text = ''
                elif isinstance(val, float):
                    text = f"{val:.3g}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)

        self._table.resizeColumnsToContents()

        # Enable buttons only if the files exist
        self._btn_qc.setEnabled(
            os.path.exists(os.path.join(output_dir, "quality_checks.pdf"))
        )
        self._btn_factsheet.setEnabled(
            os.path.exists(os.path.join(output_dir, "factsheet.pdf"))
        )
        self._btn_folder.setEnabled(os.path.isdir(output_dir))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_pdf(self, filename: str) -> None:
        """Open *filename* inside ``_output_dir`` with the system viewer."""
        if self._output_dir is None:
            return
        path = os.path.abspath(os.path.join(self._output_dir, filename))
        if not os.path.exists(path):
            logger.warning("PDF not found: %s", path)
            return
        self._open_path(path)

    def _open_folder(self) -> None:
        """Open the output folder in the system file manager."""
        if self._output_dir and os.path.isdir(self._output_dir):
            self._open_path(os.path.abspath(self._output_dir))

    @staticmethod
    def _open_path(path: str) -> None:
        """Open *path* with the OS default application."""
        try:
            if sys.platform == 'win32':
                os.startfile(path)          # pylint: disable=no-member
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as exc:
            logger.error("Could not open '%s': %s", path, exc)
