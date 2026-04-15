"""
Interactive station search and selection widget (PyQt6).

Supports all four Météo-France search modes (department, ID, name,
coordinates) and both NOAA ISD-Lite search modes (name, coordinates).
Emits a ``station_selected`` signal carrying the station row and a
normalised station_info DataFrame when the user confirms a selection.

Author:  Samy KRAIEM
Created: 2026
Updated: 2026
"""

import logging
import math

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.utils import parse_coord

logger = logging.getLogger(__name__)

# French metropolitan departments + overseas
_MF_DEPARTMENTS = (
    [str(i) for i in range(1, 96)]
    + ['971', '972', '973', '974', '975', '976', '977',
       '978', '984', '986', '987', '988', '989']
)


class StationSelectorWidget(QWidget):
    """Widget that lets the user search and select a weather station.

    Signals:
        station_selected (object, pd.DataFrame):
            Emitted when the user confirms a station.  First argument is
            the raw station identifier (str for MF, pd.Series for ISD);
            second is a normalised one-row station_info DataFrame.
    """

    station_selected = pyqtSignal(object, object, str)   # station, station_info, source

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source     = 'mf'
        self._client     = None   # mf_client.Client instance
        self._isd_client = None   # isd_client.ISDClient instance
        self._df_mf      = None   # mf_stations.csv cache
        self._results_df = None   # last search results

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_source(self, source: str, client=None, isd_client=None) -> None:
        """Switch the widget between 'mf' and 'isd' modes.

        Args:
            source (str): ``'mf'`` or ``'isd'``.
            client: Initialised ``mf_client.Client`` (required for MF).
            isd_client: Initialised ``isd_client.ISDClient`` (required for ISD).
        """
        self._source     = source
        self._client     = client
        self._isd_client = isd_client
        self._df_mf      = None   # invalidate catalogue cache on source switch
        self._results_df = None
        self._refresh_mode_visibility()
        self._clear_results()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # --- Search mode radio buttons ----------------------------------
        mode_box = QGroupBox("Search mode")
        mode_lay = QVBoxLayout(mode_box)

        self._btn_group  = QButtonGroup(self)
        self._rb_dept    = QRadioButton("By department")
        self._rb_id      = QRadioButton("By station ID")
        self._rb_name    = QRadioButton("By name")
        self._rb_coords  = QRadioButton("By coordinates")
        self._rb_map     = QRadioButton("By interactive map")
        self._rb_map.setChecked(True)

        for rb in (self._rb_dept, self._rb_id, self._rb_name, self._rb_coords, self._rb_map):
            mode_lay.addWidget(rb)
            self._btn_group.addButton(rb)

        self._btn_group.buttonClicked.connect(self._on_mode_changed)
        root.addWidget(mode_box)

        # --- Dynamic input area -----------------------------------------
        self._input_box = QGroupBox("Search parameters")
        self._input_lay = QVBoxLayout(self._input_box)
        root.addWidget(self._input_box)

        # --- Search button -----------------------------------------------
        self._btn_search = QPushButton("Search")
        self._btn_search.clicked.connect(self._do_search)
        root.addWidget(self._btn_search)

        # --- Results table -----------------------------------------------
        self._results_box = QGroupBox("Results")
        results_lay = QVBoxLayout(self._results_box)

        self._table = QTableWidget()
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table.doubleClicked.connect(self._confirm_selection)
        results_lay.addWidget(self._table)

        self._btn_select = QPushButton("Confirm selection")
        self._btn_select.setEnabled(False)
        self._btn_select.clicked.connect(self._confirm_selection)
        results_lay.addWidget(self._btn_select)

        root.addWidget(self._results_box)

        # Build initial input widgets for MF department mode
        self._rebuild_inputs()

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _refresh_mode_visibility(self) -> None:
        """Show/hide MF-only radio buttons when the source changes."""
        is_mf = (self._source == 'mf')
        self._rb_dept.setVisible(is_mf)
        self._rb_id.setVisible(is_mf)

        # If an MF-only mode was selected and we switched to ISD, reset
        if not is_mf and (self._rb_dept.isChecked() or self._rb_id.isChecked()):
            self._rb_name.setChecked(True)

        self._rebuild_inputs()

    def _on_mode_changed(self, _button) -> None:
        self._rebuild_inputs()
        self._clear_results()

    def _rebuild_inputs(self) -> None:
        """Replace the dynamic input area widgets for the active mode."""
        # Remove existing widgets
        while self._input_lay.count():
            item = self._input_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Hide/show the Search button and results table depending on mode
        is_map = self._rb_map.isChecked()
        self._btn_search.setVisible(not is_map)
        self._results_box.setVisible(not is_map)

        if self._rb_dept.isChecked() and self._source == 'mf':
            self._build_dept_inputs()
        elif self._rb_id.isChecked() and self._source == 'mf':
            self._build_id_inputs()
        elif self._rb_name.isChecked():
            self._build_name_inputs()
        elif self._rb_map.isChecked():
            self._build_map_inputs()
        else:
            self._build_coords_inputs()

    def _build_dept_inputs(self) -> None:
        from PyQt6.QtWidgets import QComboBox
        self._input_lay.addWidget(QLabel("Department:"))
        self._dept_combo = QComboBox()
        self._dept_combo.addItems(_MF_DEPARTMENTS)
        self._input_lay.addWidget(self._dept_combo)

    def _build_id_inputs(self) -> None:
        self._input_lay.addWidget(QLabel("Station ID:"))
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("e.g. 75114001")
        self._input_lay.addWidget(self._id_edit)

    def _build_name_inputs(self) -> None:
        label = "Station name:" if self._source == 'mf' else "Station name (substring):"
        self._input_lay.addWidget(QLabel(label))
        self._name_edit = QLineEdit()
        placeholder = "e.g. Paris, Lyon" if self._source == 'mf' else "e.g. Kennedy, Paris, Toulouse"
        self._name_edit.setPlaceholderText(placeholder)
        self._name_edit.returnPressed.connect(self._do_search)
        self._input_lay.addWidget(self._name_edit)

    def _build_coords_inputs(self) -> None:
        self._input_lay.addWidget(QLabel("Latitude:"))
        self._lat_edit = QLineEdit()
        self._lat_edit.setPlaceholderText("e.g. 48.85  or  48,85°N")
        self._input_lay.addWidget(self._lat_edit)

        self._input_lay.addWidget(QLabel("Longitude:"))
        self._lon_edit = QLineEdit()
        self._lon_edit.setPlaceholderText("e.g. 2.35  or  2,35°E")
        self._input_lay.addWidget(self._lon_edit)

    def _build_map_inputs(self) -> None:
        """Build the input area for the interactive-map search mode."""
        note = QLabel(
            "The map shows all available stations:\n"
            "  \u25cf Blue  — Météo-France (DPClim)\n"
            "  \u25cf Green — NOAA ISD-Lite\n\n"
            "Zoom, pan, then click a marker and press\n"
            "\"Select this station\" in the popup."
        )
        note.setStyleSheet("color: #8b949e; font-size: 11px; padding: 2px;")
        self._input_lay.addWidget(note)

        btn_open = QPushButton("Open interactive map…")
        btn_open.setStyleSheet(
            "QPushButton { background: #1f6feb; color: white; font-weight: bold;"
            "              padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background: #388bfd; }"
        )
        btn_open.clicked.connect(self._open_map_dialog)
        self._input_lay.addWidget(btn_open)

    # ------------------------------------------------------------------
    # Map dialog
    # ------------------------------------------------------------------

    def _open_map_dialog(self) -> None:
        """Load both station catalogues and open the MapDialog."""
        from src.gui.map_widget import MapDialog

        # Load MF catalogue if available
        df_mf = None
        try:
            df_mf = self._load_mf_catalogue()
        except SystemExit:
            logger.warning("mf_stations.csv not found — MF stations will not appear on map.")
        except Exception as exc:
            logger.warning("Could not load MF catalogue: %s", exc)

        # Load ISD catalogue if available
        df_isd = None
        try:
            if self._isd_client is None:
                from src.isd_client import ISDClient
                self._isd_client = ISDClient()
            df_isd = self._isd_client._load_history()
        except SystemExit:
            logger.warning("isd_stations.csv not found — ISD stations will not appear on map.")
        except Exception as exc:
            logger.warning("Could not load ISD catalogue: %s", exc)

        if df_mf is None and df_isd is None:
            QMessageBox.warning(
                self, "No catalogues",
                "Neither mf_stations.csv nor isd_stations.csv were found.\n"
                "Run fetch_mf_stations.py and/or fetch_isd_stations.py first."
            )
            return

        dlg = MapDialog(
            df_mf=df_mf,
            df_isd=df_isd,
            mf_client=self._client,
            isd_client=self._isd_client,
            parent=self,
        )

        if dlg.exec() == dlg.DialogCode.Accepted:
            result = dlg.result_data()
            if result is not None:
                station, source, station_info = result
                # Emit directly — no further resolution needed (MapDialog
                # already called get_station_info / get_station_info)
                self.station_selected.emit(station, station_info, source)

    # ------------------------------------------------------------------
    # Search dispatch
    # ------------------------------------------------------------------

    def _do_search(self) -> None:
        """Dispatch to the appropriate search method."""
        if self._rb_map.isChecked():
            return   # map mode uses its own button
        try:
            if self._rb_dept.isChecked() and self._source == 'mf':
                self._search_mf_dept()
            elif self._rb_id.isChecked() and self._source == 'mf':
                self._search_mf_id()
            elif self._rb_name.isChecked():
                if self._source == 'mf':
                    self._search_mf_name()
                else:
                    self._search_isd_name()
            else:
                if self._source == 'mf':
                    self._search_mf_coords()
                else:
                    self._search_isd_coords()
        except Exception as exc:
            QMessageBox.warning(self, "Search error", str(exc))

    # ------------------------------------------------------------------
    # MF search methods
    # ------------------------------------------------------------------

    def _search_mf_dept(self) -> None:
        if self._client is None:
            QMessageBox.warning(self, "No client", "Configure API credentials first.")
            return
        dept     = self._dept_combo.currentText()
        stations = self._client.get_stations_list(dept)
        if not stations:
            QMessageBox.information(self, "No results",
                                    f"No stations found for department {dept}.")
            return

        # Convert list of dicts to DataFrame for uniform display
        df = pd.DataFrame([{'ID': s['id'], 'Name': s['nom']} for s in stations])
        self._results_df = df
        self._populate_table(df, ['ID', 'Name'])

    def _search_mf_id(self) -> None:
        if self._client is None:
            QMessageBox.warning(self, "No client", "Configure API credentials first.")
            return
        station_id = self._id_edit.text().strip()
        if not station_id:
            QMessageBox.warning(self, "Input required", "Please enter a station ID.")
            return
        info = self._client.get_station_info(station_id)
        if info is None or info.empty:
            QMessageBox.information(self, "Not found",
                                    f"Station '{station_id}' not found.")
            return
        df = pd.DataFrame({'ID': [station_id], 'Name': [info.iloc[0]['Nom']]})
        self._results_df = df
        self._populate_table(df, ['ID', 'Name'])

    def _load_mf_catalogue(self) -> pd.DataFrame:
        """Load and cache mf_stations.csv."""
        if self._df_mf is not None:
            return self._df_mf
        from src.station_selector import load_mf_stations
        self._df_mf = load_mf_stations()
        return self._df_mf

    def _search_mf_name(self) -> None:
        query = self._name_edit.text().strip()
        if not query:
            QMessageBox.warning(self, "Input required", "Please enter a station name.")
            return
        df   = self._load_mf_catalogue()
        hits = df[df['Nom'].str.contains(query, case=False, na=False)].head(30).reset_index(drop=True)
        if hits.empty:
            QMessageBox.information(self, "No results", f"No station matches '{query}'.")
            return
        self._results_df = hits
        cols = ['ID', 'Nom', 'Altitude', 'DateDebut', 'DateFin']
        self._populate_table(hits, [c for c in cols if c in hits.columns])

    def _search_mf_coords(self) -> None:
        lat, lon = self._parse_coords()
        if lat is None:
            return
        df      = self._load_mf_catalogue().dropna(subset=['Latitude', 'Longitude'])
        nearest = self._haversine_nearest(df, lat, lon, 'Latitude', 'Longitude')
        self._results_df = nearest
        cols = ['ID', 'Nom', 'Altitude', 'DateDebut', 'DateFin', 'distance_km']
        self._populate_table(nearest, [c for c in cols if c in nearest.columns])

    # ------------------------------------------------------------------
    # ISD search methods
    # ------------------------------------------------------------------

    def _search_isd_name(self) -> None:
        query = self._name_edit.text().strip()
        if not query:
            QMessageBox.warning(self, "Input required", "Please enter a station name.")
            return
        if self._isd_client is None:
            from src.isd_client import ISDClient
            self._isd_client = ISDClient()
        hits = self._isd_client.search_stations(query)
        if hits.empty:
            QMessageBox.information(self, "No results", f"No station matches '{query}'.")
            return
        self._results_df = hits
        cols = ['STATION NAME', 'CTRY', 'LAT', 'LON', 'ELEV(M)', 'BEGIN', 'END']
        self._populate_table(hits, [c for c in cols if c in hits.columns])

    def _search_isd_coords(self) -> None:
        lat, lon = self._parse_coords()
        if lat is None:
            return
        if self._isd_client is None:
            from src.isd_client import ISDClient
            self._isd_client = ISDClient()
        nearest = self._isd_client.find_nearest_stations(lat, lon, n=10)
        self._results_df = nearest
        cols = ['STATION NAME', 'CTRY', 'LAT', 'LON', 'ELEV(M)', 'BEGIN', 'END', 'distance_km']
        self._populate_table(nearest, [c for c in cols if c in nearest.columns])

    # ------------------------------------------------------------------
    # Helper: coordinate parsing
    # ------------------------------------------------------------------

    def _parse_coords(self):
        """Parse and validate lat/lon inputs.  Returns (lat, lon) or (None, None)."""
        try:
            lat = parse_coord(self._lat_edit.text().strip())
            lon = parse_coord(self._lon_edit.text().strip())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid coordinates", str(exc))
            return None, None
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            QMessageBox.warning(self, "Out of range",
                                "Coordinates are out of the valid range.")
            return None, None
        return lat, lon

    # ------------------------------------------------------------------
    # Helper: Haversine nearest-neighbours
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_nearest(
        df: pd.DataFrame,
        lat: float,
        lon: float,
        lat_col: str,
        lon_col: str,
        n: int = 10,
    ) -> pd.DataFrame:
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        dlat  = np.radians(df[lat_col].values) - lat_r
        dlon  = np.radians(df[lon_col].values) - lon_r
        a     = (
            np.sin(dlat / 2) ** 2
            + math.cos(lat_r)
            * np.cos(np.radians(df[lat_col].values))
            * np.sin(dlon / 2) ** 2
        )
        dist = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
        result = df.copy()
        result['distance_km'] = dist
        return result.nsmallest(n, 'distance_km').reset_index(drop=True)

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _populate_table(self, df: pd.DataFrame, cols: list) -> None:
        """Fill the results QTableWidget from a DataFrame subset."""
        show = df[cols]
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(show))

        for r, (_, row) in enumerate(show.iterrows()):
            for c, col in enumerate(cols):
                val  = row[col]
                text = f"{val:.1f}" if col == 'distance_km' and pd.notna(val) else (
                    str(val) if pd.notna(val) else ''
                )
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)

        self._btn_select.setEnabled(True)

    def _clear_results(self) -> None:
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._btn_select.setEnabled(False)
        self._results_df = None

    # ------------------------------------------------------------------
    # Station confirmation
    # ------------------------------------------------------------------

    def _confirm_selection(self) -> None:
        """Emit ``station_selected`` with the highlighted table row."""
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No selection",
                                    "Please click a row in the results table first.")
            return

        if self._results_df is None or row >= len(self._results_df):
            return

        selected = self._results_df.iloc[row]

        if self._source == 'mf':
            self._confirm_mf(selected)
        else:
            self._confirm_isd(selected)

    def _confirm_mf(self, selected: pd.Series) -> None:
        """Resolve MF station ID → station_info DataFrame and emit signal."""
        station_id = str(selected['ID'])
        if self._client is None:
            QMessageBox.warning(self, "No client", "Configure API credentials first.")
            return
        station_info = self._client.get_station_info(station_id)
        if station_info is None or station_info.empty:
            QMessageBox.warning(self, "Error",
                                f"Could not retrieve info for station '{station_id}'.")
            return
        self.station_selected.emit(station_id, station_info, 'mf')

    def _confirm_isd(self, selected: pd.Series) -> None:
        """Build ISD station_info DataFrame and emit signal."""
        if self._isd_client is None:
            from src.isd_client import ISDClient
            self._isd_client = ISDClient()
        station_info = self._isd_client.get_station_info(selected)
        self.station_selected.emit(selected, station_info, 'isd')
