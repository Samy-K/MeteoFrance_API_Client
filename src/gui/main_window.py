"""
Main application window for MeteoFrance API Client (PyQt6).

Orchestrates the full GUI workflow:
  1. Source selection (Météo-France / NOAA ISD)
  2. API configuration (MF credentials dialog)
  3. Station search and selection
  4. Year-range input
  5. Background download via DownloadWorker (QThread)
  6. Results display (dataset preview + PDF report buttons)

Logging output from all modules is captured and displayed in a live
log panel on the right-hand side of the window.

Author:  Samy KRAIEM
Created: 2026
Updated: 2026
"""

import configparser
import logging
import os

import pandas as pd
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.download_worker import DownloadWorker
from src.gui.results_widget import ResultsWidget
from src.gui.station_selector_widget import StationSelectorWidget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qt logging handler — thread-safe bridge from logging → QTextEdit
# ---------------------------------------------------------------------------

from PyQt6.QtCore import QObject, pyqtSignal as _Signal


class _LogSignalRelay(QObject):
    """Internal QObject used to ferry log records across thread boundaries."""
    record = _Signal(str, int)


class QtLogHandler(logging.Handler):
    """logging.Handler that appends coloured records to a QTextEdit.

    Because logging.Handler.emit() can be called from any thread, the
    actual widget update is deferred through a Qt signal so it is
    always executed on the main thread.

    Args:
        text_widget (QTextEdit): Target widget for log output.
    """

    _LEVEL_COLOURS = {
        logging.DEBUG:    '#6cb6ff',   # blue
        logging.INFO:     '#57ab5a',   # green
        logging.WARNING:  '#d29922',   # amber
        logging.ERROR:    '#e5534b',   # red
        logging.CRITICAL: '#ff7b72',   # bright red
    }

    def __init__(self, text_widget: QTextEdit):
        super().__init__()
        self._widget  = text_widget
        self._relay   = _LogSignalRelay()
        self._relay.record.connect(self._append)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                      datefmt='%H:%M:%S')
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Forward *record* to the main thread via a Qt signal."""
        try:
            msg = self.format(record)
            self._relay.record.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)

    def _append(self, msg: str, levelno: int) -> None:
        """Append *msg* to the QTextEdit with level-appropriate colour."""
        colour = self._LEVEL_COLOURS.get(levelno, '#cccccc')
        fmt    = QTextCharFormat()
        fmt.setForeground(QColor(colour))

        cursor = self._widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(msg + '\n', fmt)

        # Auto-scroll to the bottom
        self._widget.setTextCursor(cursor)
        self._widget.ensureCursorVisible()


# ---------------------------------------------------------------------------
# API credentials dialog
# ---------------------------------------------------------------------------

class ApiConfigDialog(QDialog):
    """Modal dialog for editing Météo-France API credentials.

    Reads ``API_config.txt`` on open and saves the edited values back on
    acceptance.

    Args:
        parent: Qt parent widget.
    """

    _CONFIG_FILE = 'API_config.txt'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Météo-France API Configuration")
        self.setMinimumWidth(480)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._app_id = QLineEdit()
        self._token  = QLineEdit()
        self._server = QLineEdit()
        self._app_id.setPlaceholderText("Base-64 OAuth2 client credentials")
        self._token.setPlaceholderText("Static API token")
        self._server.setText("https://public-api.meteofrance.fr/public/DPClim/v1")

        form.addRow("APPLICATION_ID:", self._app_id)
        form.addRow("TOKEN:", self._token)
        form.addRow("DATA_SERVER:", self._server)
        layout.addLayout(form)

        note = QLabel(
            "Provide at least one of APPLICATION_ID or TOKEN.\n"
            "Credentials are saved to API_config.txt (not committed to git)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load(self) -> None:
        """Populate fields from API_config.txt if it exists."""
        cfg = configparser.ConfigParser()
        cfg.read(self._CONFIG_FILE)
        if cfg.has_section('Parameters'):
            self._app_id.setText(cfg.get('Parameters', 'APPLICATION_ID', fallback=''))
            self._token.setText(cfg.get('Parameters', 'TOKEN', fallback=''))
            self._server.setText(
                cfg.get('Parameters', 'DATA_SERVER',
                         fallback='https://public-api.meteofrance.fr/public/DPClim/v1')
            )

    def _save(self) -> None:
        """Write current field values to API_config.txt and accept the dialog."""
        cfg = configparser.ConfigParser()
        cfg['Parameters'] = {
            'APPLICATION_ID': self._app_id.text().strip(),
            'DATA_SERVER':    self._server.text().strip(),
            'TOKEN':          self._token.text().strip(),
        }
        with open(self._CONFIG_FILE, 'w') as fh:
            cfg.write(fh)
        logger.info("API credentials saved to %s.", self._CONFIG_FILE)
        self.accept()

    def get_config(self) -> dict:
        """Return the current field values as a dict.

        Returns:
            dict: Keys ``APPLICATION_ID``, ``TOKEN``, ``DATA_SERVER``.
        """
        return {
            'APPLICATION_ID': self._app_id.text().strip(),
            'TOKEN':          self._token.text().strip(),
            'DATA_SERVER':    self._server.text().strip(),
        }


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Top-level window orchestrating the full GUI workflow.

    Layout (horizontal splitter):
      Left  — configuration panel (source, station search, year range, download)
      Right — activity log + results panel
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeteoFrance API Client")
        self.resize(1280, 800)

        self._worker         = None           # active DownloadWorker, if any
        self._station        = None           # selected station (str or pd.Series)
        self._station_info   = None           # pd.DataFrame
        self._api_config     = {}             # latest MF credentials

        self._build_ui()
        self._install_log_handler()
        self._on_source_changed(0)            # initialise to MF mode

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([480, 780])

    # --- Left panel -------------------------------------------------------

    def _build_left_panel(self) -> QScrollArea:
        """Return a scrollable left panel with all configuration controls."""
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setSpacing(8)

        # Source selection + API config button
        src_box = QGroupBox("Data source")
        src_lay = QHBoxLayout(src_box)
        self._src_combo = QComboBox()
        self._src_combo.addItems(["Météo-France (DPClim)", "NOAA ISD-Lite"])
        self._src_combo.currentIndexChanged.connect(self._on_source_changed)
        src_lay.addWidget(self._src_combo, stretch=1)

        self._btn_api_cfg = QPushButton("Configure API…")
        self._btn_api_cfg.clicked.connect(self._open_api_config)
        src_lay.addWidget(self._btn_api_cfg)
        lay.addWidget(src_box)

        # Station selector
        self._station_selector = StationSelectorWidget()
        self._station_selector.station_selected.connect(self._on_station_selected)
        lay.addWidget(self._station_selector)

        # Selected station info banner
        self._selected_label = QLabel("No station selected.")
        self._selected_label.setWordWrap(True)
        self._selected_label.setStyleSheet(
            "background: #1e3a1e; color: #8bc18b; padding: 6px; border-radius: 4px;"
        )
        lay.addWidget(self._selected_label)

        # Year range
        year_box = QGroupBox("Year range")
        year_lay = QFormLayout(year_box)
        self._spin_start = QSpinBox()
        self._spin_end   = QSpinBox()
        for spin in (self._spin_start, self._spin_end):
            spin.setRange(1900, 2100)
            spin.setValue(2020)
        self._spin_end.setValue(2024)
        year_lay.addRow("Start year:", self._spin_start)
        year_lay.addRow("End year:",   self._spin_end)
        lay.addWidget(year_box)

        # Download button
        self._btn_download = QPushButton("Download && generate reports")
        self._btn_download.setEnabled(False)
        self._btn_download.setStyleSheet(
            "QPushButton { background: #1f6feb; color: white; font-weight: bold;"
            "              padding: 8px; border-radius: 4px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self._btn_download.clicked.connect(self._start_download)
        lay.addWidget(self._btn_download)

        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    # --- Right panel ------------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        """Return the right panel containing the log and results widgets."""
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Activity log
        log_box = QGroupBox("Activity log")
        log_lay = QVBoxLayout(log_box)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Consolas", 9))
        self._log_text.setStyleSheet("background: #0d1117; color: #c9d1d9;")
        self._log_text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._log_text.setMinimumHeight(180)
        log_lay.addWidget(self._log_text)

        btn_clear = QPushButton("Clear log")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._log_text.clear)
        log_lay.addWidget(btn_clear, alignment=Qt.AlignmentFlag.AlignRight)
        lay.addWidget(log_box, stretch=1)

        # Results panel (hidden until a download completes)
        results_box = QGroupBox("Results")
        results_lay = QVBoxLayout(results_box)
        self._results_widget = ResultsWidget()
        results_lay.addWidget(self._results_widget)
        self._results_box = results_box
        self._results_box.setVisible(False)
        lay.addWidget(results_box, stretch=2)

        return panel

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------

    def _install_log_handler(self) -> None:
        """Attach the Qt log handler to the root logger."""
        handler = QtLogHandler(self._log_text)
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_source_changed(self, index: int) -> None:
        """Reinitialise the station selector for the newly selected source."""
        source = 'mf' if index == 0 else 'isd'
        self._btn_api_cfg.setVisible(source == 'mf')
        self._station = None
        self._station_info = None
        self._btn_download.setEnabled(False)
        self._selected_label.setText("No station selected.")

        if source == 'mf':
            client = self._build_mf_client()
            self._station_selector.set_source('mf', client=client)
        else:
            from src.isd_client import ISDClient
            self._station_selector.set_source('isd', isd_client=ISDClient())

    def _open_api_config(self) -> None:
        """Open the MF API credentials dialog and reinitialise the client."""
        dlg = ApiConfigDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._api_config = dlg.get_config()
            client = self._build_mf_client()
            self._station_selector.set_source('mf', client=client)

    def _build_mf_client(self):
        """Build a Météo-France Client from the stored or on-disk credentials.

        Returns:
            Client or None if no valid credentials are available.
        """
        from src.mf_client import Client

        # Merge on-disk credentials with any runtime overrides
        cfg = configparser.ConfigParser()
        cfg.read('API_config.txt')
        if cfg.has_section('Parameters'):
            disk = {
                'APPLICATION_ID': cfg.get('Parameters', 'APPLICATION_ID', fallback=''),
                'TOKEN':          cfg.get('Parameters', 'TOKEN', fallback=''),
                'DATA_SERVER':    cfg.get('Parameters', 'DATA_SERVER',
                                          fallback='https://public-api.meteofrance.fr/public/DPClim/v1'),
            }
            # Runtime overrides take precedence
            for k, v in self._api_config.items():
                if v:
                    disk[k] = v
            self._api_config = disk

        token  = self._api_config.get('TOKEN', '').replace('"', '').strip()
        app_id = self._api_config.get('APPLICATION_ID', '').strip()
        server = self._api_config.get('DATA_SERVER', '').strip()

        if not token and not app_id:
            logger.warning(
                "No API credentials found. "
                "Use 'Configure API…' to enter your Météo-France credentials."
            )
            return None

        try:
            return Client(api_key=token or None,
                          application_id=app_id or None,
                          base_url=server or None)
        except Exception as exc:
            logger.error("Failed to initialise MF client: %s", exc)
            return None

    def _on_station_selected(self, station, station_info: pd.DataFrame, source: str = '') -> None:
        """Handle a station being confirmed in the selector widget.

        Args:
            station: Raw station identifier (str for MF, pd.Series for ISD).
            station_info: One-row normalised station DataFrame.
            source: ``'mf'`` or ``'isd'``.  When provided by the map dialog,
                the source combo is updated silently so the download pipeline
                uses the correct backend even if the user had not switched it
                manually.
        """
        self._station      = station
        self._station_info = station_info

        # Sync the source combo to the actual source of the selected station
        # without triggering _on_source_changed (which would reset the selection).
        if source in ('mf', 'isd'):
            target_index = 0 if source == 'mf' else 1
            if self._src_combo.currentIndex() != target_index:
                self._src_combo.blockSignals(True)
                self._src_combo.setCurrentIndex(target_index)
                self._src_combo.blockSignals(False)
                # Keep the API-config button in sync with the new source
                self._btn_api_cfg.setVisible(source == 'mf')

        row = station_info.iloc[0]

        # --- Formatted fields -----------------------------------------------
        name = str(row.get('Nom', '—'))
        sid  = str(row.get('ID',  '—'))

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

        source_label = self._src_combo.currentText()

        # --- Rich-text banner -----------------------------------------------
        sep = '<span style="color:#444;">&nbsp;|&nbsp;</span>'
        self._selected_label.setText(
            f'<b style="color:#8bc18b">{name}</b>'
            f'&nbsp;&nbsp;<span style="color:#6e8c6e">({sid})</span>'
            f'<br/>'
            f'<span style="color:#6e8c6e">lat</span>&nbsp;{lat}'
            f'{sep}'
            f'<span style="color:#6e8c6e">lon</span>&nbsp;{lon}'
            f'{sep}'
            f'<span style="color:#6e8c6e">alt</span>&nbsp;{alt}'
            f'<br/>'
            f'<span style="color:#6e8c6e">Period:</span>&nbsp;'
            f'{date_debut}&nbsp;&ndash;&nbsp;{date_fin}'
            f'{sep}'
            f'<span style="color:#6e8c6e">Source:</span>&nbsp;{source_label}'
        )
        self._selected_label.setTextFormat(Qt.TextFormat.RichText)
        self._btn_download.setEnabled(True)

        # --- Pre-fill year spinboxes from station metadata ------------------
        try:
            y_start = int(str(date_debut)[:4])
        except (ValueError, TypeError):
            y_start = 2000
        try:
            y_end = int(str(date_fin_raw)[:4]) if pd.notna(date_fin_raw) else self._spin_end.maximum()
        except (ValueError, TypeError):
            y_end = 2024
        self._spin_start.setValue(max(1900, y_start))
        self._spin_end.setValue(min(2100, y_end if y_end > 1900 else 2024))

        logger.info(
            "Station selected — %s (%s)  lat %s  lon %s  %s – %s  [%s]",
            name, sid, lat, lon, date_debut, date_fin, source_label,
        )

    def _start_download(self) -> None:
        """Validate inputs and launch the DownloadWorker."""
        if self._station is None or self._station_info is None:
            QMessageBox.warning(self, "No station", "Please select a station first.")
            return

        start = self._spin_start.value()
        end   = self._spin_end.value()
        if end < start:
            QMessageBox.warning(self, "Invalid range",
                                "End year must be ≥ start year.")
            return

        source = 'mf' if self._src_combo.currentIndex() == 0 else 'isd'

        logger.info(
            "Starting download — source: %s | period: %d–%d", source, start, end
        )

        self._btn_download.setEnabled(False)
        self._btn_download.setText("Downloading…")
        self._results_box.setVisible(False)

        self._worker = DownloadWorker(
            source=source,
            station=self._station,
            station_info=self._station_info,
            start_year=start,
            end_year=end,
            api_config=self._api_config,
            parent=self,
        )
        self._worker.finished.connect(self._on_download_finished)
        self._worker.error.connect(self._on_download_error)
        self._worker.start()

    def _on_download_finished(
        self,
        dataset,
        station_info: pd.DataFrame,
        start_year: int,
        end_year: int,
        output_dir: str,
    ) -> None:
        """Populate the results panel and re-enable the download button."""
        logger.info("Pipeline complete — output in '%s'.", output_dir)
        self._btn_download.setEnabled(True)
        self._btn_download.setText("Download && generate reports")

        self._results_widget.populate(
            dataset, station_info, start_year, end_year, output_dir,
            source_label=self._src_combo.currentText(),
        )
        self._results_box.setVisible(True)

    def _on_download_error(self, message: str) -> None:
        """Show an error dialog and re-enable the download button."""
        logger.error("Download failed: %s", message)
        self._btn_download.setEnabled(True)
        self._btn_download.setText("Download && generate reports")
        QMessageBox.critical(self, "Download error", message)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Ensure the worker thread is stopped before the window closes."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(3000)
        event.accept()
