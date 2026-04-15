"""
Interactive Leaflet.js map widget for station selection (PyQt6).

Embeds a full Leaflet map inside a QWebEngineView.  Météo-France
stations are rendered in blue (#0078d4) and NOAA ISD-Lite stations
in green (#2ea043).  Marker clustering is applied so the map stays
responsive even with 30 000+ ISD markers.

Clicking a marker opens a popup; the "Select this station" button
inside the popup sends the station ID and source back to Python via
QWebChannel, closing the selection dialog.

Public entry-point:
    dialog = MapDialog(df_mf=..., df_isd=..., parent=...)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        station_id, source, station_info = dialog.result_data()

Author:  Samy KRAIEM
Created: 2026
Updated: 2026
"""

import json
import logging
import os
import tempfile

import pandas as pd
from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JS → Python bridge
# ---------------------------------------------------------------------------

class _Bridge(QObject):
    """QObject registered with QWebChannel and callable from JavaScript.

    The Leaflet popup's 'Select' button calls
    ``bridge.on_station_clicked(id, source)`` which triggers
    the ``station_clicked`` signal on the main thread.
    """

    station_clicked = pyqtSignal(str, str)   # (station_id, source)

    @pyqtSlot(str, str)
    def on_station_clicked(self, station_id: str, source: str) -> None:
        """Slot invoked by JavaScript when the user confirms a station.

        Args:
            station_id: Station identifier (MF ID string or ISD
                ``USAF-WBAN`` composite key).
            source: ``'mf'`` or ``'isd'``.
        """
        logger.debug("Map selection: %s (%s)", station_id, source)
        self.station_clicked.emit(station_id, source)


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

def _make_html(
    mf_json: str,
    isd_json: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    mf_count: int,
    isd_count: int,
) -> str:
    """Return the full Leaflet HTML page as a string.

    Uses plain string replacement (not an f-string) to avoid conflicts
    between Python's ``{}`` interpolation and JavaScript object literal
    syntax.

    Args:
        mf_json:    JSON array of MF station dicts.
        isd_json:   JSON array of ISD station dicts.
        center_lat: Initial map centre latitude.
        center_lon: Initial map centre longitude.
        zoom:       Initial Leaflet zoom level.
        mf_count:   Number of MF markers (shown in layer-control label).
        isd_count:  Number of ISD markers (shown in layer-control label).

    Returns:
        str: Self-contained HTML page.
    """
    template = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Station Map</title>
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0d1117; }
    #map { width: 100%; height: 100vh; }

    /* Dark popup shell */
    .dark-popup .leaflet-popup-content-wrapper {
      background: #161b22;
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 8px;
      box-shadow: 0 6px 24px rgba(0,0,0,.6);
      padding: 0;
    }
    .dark-popup .leaflet-popup-content { margin: 0; }
    .dark-popup .leaflet-popup-tip { background: #161b22; }
    .dark-popup a.leaflet-popup-close-button {
      color: #8b949e; top: 6px; right: 8px;
    }

    /* Popup content */
    .popup-inner { padding: 10px 14px 12px; min-width: 180px; }
    .popup-title {
      font-weight: 700; font-size: 13px; margin-bottom: 6px;
      padding-bottom: 4px; border-bottom: 1px solid #30363d;
    }
    .popup-row { font-size: 11px; color: #8b949e; margin: 3px 0; }
    .popup-row span { color: #c9d1d9; }
    .btn-select {
      margin-top: 10px; padding: 6px 0; width: 100%;
      background: #238636; color: #fff; border: none;
      border-radius: 5px; cursor: pointer; font-size: 12px;
      font-weight: 600; letter-spacing: .3px; transition: background .15s;
    }
    .btn-select:hover { background: #2ea043; }

    /* Layer control — dark theme */
    .leaflet-control-layers {
      background: #161b22 !important;
      border: 1px solid #30363d !important;
      color: #c9d1d9 !important;
      border-radius: 6px !important;
    }
    .leaflet-control-layers-toggle { background-color: #1f6feb; }
    .leaflet-control-layers label { color: #c9d1d9; }

    /* Legend */
    .legend {
      background: #161b22;
      color: #c9d1d9;
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid #30363d;
      font-size: 12px;
      line-height: 1.6;
    }
    .legend-item { display: flex; align-items: center; gap: 8px; }
    .dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  </style>
</head>
<body>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>

  <script>
    /* ------------------------------------------------------------------ */
    /* Station data injected by Python                                      */
    /* ------------------------------------------------------------------ */
    const MF_STATIONS  = __MF_JSON__;
    const ISD_STATIONS = __ISD_JSON__;

    /* ------------------------------------------------------------------ */
    /* Map initialisation — dark CartoDB tiles                              */
    /* ------------------------------------------------------------------ */
    const map = L.map('map', { preferCanvas: true })
      .setView([__CENTER_LAT__, __CENTER_LON__], __ZOOM__);

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
          + '&copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
      }
    ).addTo(map);

    /* ------------------------------------------------------------------ */
    /* QWebChannel bridge                                                   */
    /* ------------------------------------------------------------------ */
    let bridge = null;
    if (typeof QWebChannel !== 'undefined') {
      new QWebChannel(qt.webChannelTransport, ch => {
        bridge = ch.objects.bridge;
      });
    }

    /* Called from popup button */
    function selectStation(id, source) {
      if (bridge) bridge.on_station_clicked(id, source);
    }

    /* ------------------------------------------------------------------ */
    /* Shared helpers                                                        */
    /* ------------------------------------------------------------------ */
    function dotIcon(color) {
      return L.divIcon({
        className: '',
        html: '<div style="width:10px;height:10px;border-radius:50%;'
            + 'background:' + color + ';border:2px solid rgba(255,255,255,.75);'
            + 'box-shadow:0 1px 4px rgba(0,0,0,.55);"></div>',
        iconSize:   [14, 14],
        iconAnchor: [7, 7],
      });
    }

    function clusterIcon(color) {
      return function(cluster) {
        return L.divIcon({
          className: '',
          html: '<div style="background:' + color + ';color:#fff;border-radius:50%;'
              + 'width:32px;height:32px;display:flex;align-items:center;'
              + 'justify-content:center;font-size:11px;font-weight:700;'
              + 'border:2px solid rgba(255,255,255,.55);">'
              + cluster.getChildCount() + '</div>',
          iconSize: [32, 32],
        });
      };
    }

    function buildPopup(s, source, color) {
      var altStr   = (s.alt != null) ? s.alt + ' m' : '—';
      var safeId   = s.id.replace(/'/g, "\\'");
      return '<div class="popup-inner">'
           + '<div class="popup-title" style="color:' + color + '">' + s.name + '</div>'
           + '<div class="popup-row">ID:&nbsp;&nbsp;&nbsp;&nbsp;<span>' + s.id + '</span></div>'
           + '<div class="popup-row">Altitude: <span>' + altStr + '</span></div>'
           + '<div class="popup-row">Period:&nbsp; <span>' + s.begin + ' – ' + s.end + '</span></div>'
           + '<button class="btn-select" '
           + 'onclick="selectStation(\'' + safeId + '\',\'' + source + '\')">'
           + '&#10003;&nbsp; Select this station'
           + '</button></div>';
    }

    /* ------------------------------------------------------------------ */
    /* Météo-France layer — blue                                            */
    /* ------------------------------------------------------------------ */
    var MF_COLOR   = '#0078d4';
    var mfIcon     = dotIcon(MF_COLOR);
    var mfCluster  = L.markerClusterGroup({
      maxClusterRadius: 50,
      iconCreateFunction: clusterIcon(MF_COLOR),
    });
    MF_STATIONS.forEach(function(s) {
      var m = L.marker([s.lat, s.lon], { icon: mfIcon });
      m.bindPopup(buildPopup(s, 'mf', MF_COLOR), { className: 'dark-popup', maxWidth: 240 });
      mfCluster.addLayer(m);
    });
    map.addLayer(mfCluster);

    /* ------------------------------------------------------------------ */
    /* NOAA ISD layer — green                                               */
    /* ------------------------------------------------------------------ */
    var ISD_COLOR  = '#2ea043';
    var isdIcon    = dotIcon(ISD_COLOR);
    var isdCluster = L.markerClusterGroup({
      maxClusterRadius: 50,
      iconCreateFunction: clusterIcon(ISD_COLOR),
    });
    ISD_STATIONS.forEach(function(s) {
      var m = L.marker([s.lat, s.lon], { icon: isdIcon });
      m.bindPopup(buildPopup(s, 'isd', ISD_COLOR), { className: 'dark-popup', maxWidth: 240 });
      isdCluster.addLayer(m);
    });
    map.addLayer(isdCluster);

    /* ------------------------------------------------------------------ */
    /* Layer control                                                        */
    /* ------------------------------------------------------------------ */
    var overlays = {};
    overlays['<span style="color:' + MF_COLOR  + '">&#11044;</span>&nbsp;'
           + 'Météo-France (__MF_COUNT__)']  = mfCluster;
    overlays['<span style="color:' + ISD_COLOR + '">&#11044;</span>&nbsp;'
           + 'NOAA ISD-Lite (__ISD_COUNT__)'] = isdCluster;
    L.control.layers(null, overlays, { collapsed: false }).addTo(map);

    /* ------------------------------------------------------------------ */
    /* Legend                                                               */
    /* ------------------------------------------------------------------ */
    var legend = L.control({ position: 'bottomleft' });
    legend.onAdd = function() {
      var div = L.DomUtil.create('div', 'legend');
      div.innerHTML =
        '<div class="legend-item"><div class="dot" style="background:' + MF_COLOR  + '"></div>Météo-France</div>'
      + '<div class="legend-item"><div class="dot" style="background:' + ISD_COLOR + '"></div>NOAA ISD-Lite</div>';
      return div;
    };
    legend.addTo(map);
  </script>
</body>
</html>"""

    return (
        template
        .replace('__MF_JSON__',    mf_json)
        .replace('__ISD_JSON__',   isd_json)
        .replace('__CENTER_LAT__', str(center_lat))
        .replace('__CENTER_LON__', str(center_lon))
        .replace('__ZOOM__',       str(zoom))
        .replace('__MF_COUNT__',   str(mf_count))
        .replace('__ISD_COUNT__',  str(isd_count))
    )


# ---------------------------------------------------------------------------
# MapWidget — QWebEngineView wrapper
# ---------------------------------------------------------------------------

class MapWidget(QWidget):
    """Leaflet map embedded in a PyQt6 widget.

    Loads station data into an HTML page served via ``setHtml`` and
    bridges JavaScript click events back to Python through QWebChannel.

    Signals:
        station_clicked (str, str): Emitted when the user clicks
            "Select this station" in a popup.  Arguments are
            ``station_id`` and ``source`` (``'mf'`` or ``'isd'``).
    """

    station_clicked = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bridge    = _Bridge()
        self._bridge.station_clicked.connect(self.station_clicked)

        self._channel = QWebChannel()
        self._channel.registerObject('bridge', self._bridge)

        self._view = QWebEngineView()
        self._view.page().setWebChannel(self._channel)

        # Allow file:// origin to load CDN resources (Leaflet, MarkerCluster)
        self._view.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )

        self._temp_html: str | None = None   # path of the current temp HTML file

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    def load_stations(
        self,
        df_mf: pd.DataFrame = None,
        df_isd: pd.DataFrame = None,
        center_lat: float = 20.0,
        center_lon: float = 0.0,
        zoom: int = 2,
    ) -> None:
        """Render the map with the provided station datasets.

        Writes the generated HTML to a temporary file and loads it via a
        ``file://`` URL.  This is required because ``setHtml()`` uses
        ``about:blank`` as the security origin and blocks CDN resources.
        The ``LocalContentCanAccessRemoteUrls`` setting is enabled on the
        page so that Leaflet and MarkerCluster can load from unpkg.com.

        Args:
            df_mf: Météo-France station catalogue (from mf_stations.csv).
            df_isd: NOAA ISD station catalogue (from isd_stations.csv).
            center_lat: Initial map centre latitude.
            center_lon: Initial map centre longitude.
            zoom: Initial Leaflet zoom level.
        """
        mf_data  = _df_mf_to_list(df_mf)   if df_mf  is not None else []
        isd_data = _df_isd_to_list(df_isd) if df_isd is not None else []

        logger.info(
            "Loading map — %d MF stations, %d ISD stations.",
            len(mf_data), len(isd_data),
        )

        html = _make_html(
            mf_json=json.dumps(mf_data, ensure_ascii=False),
            isd_json=json.dumps(isd_data, ensure_ascii=False),
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=zoom,
            mf_count=len(mf_data),
            isd_count=len(isd_data),
        )

        # Clean up previous temp file
        if self._temp_html and os.path.exists(self._temp_html):
            try:
                os.unlink(self._temp_html)
            except OSError:
                pass

        # Write to a temp file — file:// URL gives a real origin so CDN loads
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8'
        ) as fh:
            fh.write(html)
            self._temp_html = fh.name

        self._view.load(QUrl.fromLocalFile(self._temp_html))


# ---------------------------------------------------------------------------
# Data-conversion helpers
# ---------------------------------------------------------------------------

def _df_mf_to_list(df: pd.DataFrame) -> list:
    """Convert a Météo-France station catalogue DataFrame to JSON-ready dicts.

    Stations that share identical coordinates (rounded to 4 decimal places,
    ~10 m precision) are deduplicated: the most recent station is kept
    (active stations — DateFin NaN — take priority, then latest DateFin).

    Args:
        df (pd.DataFrame): Rows from mf_stations.csv.

    Returns:
        list: Dicts with keys ``id``, ``name``, ``lat``, ``lon``, ``alt``,
            ``begin``, ``end``.
    """
    valid = df.dropna(subset=['Latitude', 'Longitude']).copy()

    # Sort: active stations (DateFin NaN) first, then by DateFin descending
    valid['_sort'] = pd.to_datetime(valid['DateFin'], errors='coerce')
    valid.sort_values('_sort', ascending=False, na_position='first', inplace=True)

    # Deduplicate by position — keep first occurrence (= most recent)
    valid['_lat_r'] = valid['Latitude'].round(4)
    valid['_lon_r'] = valid['Longitude'].round(4)
    before = len(valid)
    valid.drop_duplicates(subset=['_lat_r', '_lon_r'], keep='first', inplace=True)
    dropped = before - len(valid)
    if dropped:
        logger.debug("MF map: removed %d duplicate position(s).", dropped)

    result = []
    for _, row in valid.iterrows():
        date_fin_raw = row.get('DateFin')
        date_fin = str(date_fin_raw)[:10] if pd.notna(date_fin_raw) else ''
        result.append({
            'id':    str(row.get('ID', '')),
            'name':  str(row.get('Nom', '')),
            'lat':   round(float(row['Latitude']),  5),
            'lon':   round(float(row['Longitude']), 5),
            'alt':   round(float(row['Altitude']), 1)
                     if pd.notna(row.get('Altitude')) else None,
            'begin': str(row.get('DateDebut', ''))[:10],
            'end':   date_fin if date_fin else 'present',
        })
    return result


def _df_isd_to_list(df: pd.DataFrame) -> list:
    """Convert a NOAA ISD station catalogue DataFrame to JSON-ready dicts.

    Stations that share identical coordinates (rounded to 4 decimal places)
    are deduplicated: the station with the most recent END date is kept.

    Args:
        df (pd.DataFrame): Rows from isd_stations.csv.

    Returns:
        list: Dicts with keys ``id``, ``name``, ``lat``, ``lon``, ``alt``,
            ``begin``, ``end``.  The ``id`` field is ``USAF-WBAN``.
    """
    valid = df.dropna(subset=['LAT', 'LON']).copy()

    # Sort by END date descending so we keep the most recent station
    valid['_end_num'] = pd.to_numeric(valid['END'], errors='coerce')
    valid.sort_values('_end_num', ascending=False, na_position='last', inplace=True)

    # Deduplicate by position
    valid['_lat_r'] = valid['LAT'].round(4)
    valid['_lon_r'] = valid['LON'].round(4)
    before = len(valid)
    valid.drop_duplicates(subset=['_lat_r', '_lon_r'], keep='first', inplace=True)
    dropped = before - len(valid)
    if dropped:
        logger.debug("ISD map: removed %d duplicate position(s).", dropped)

    result = []
    for _, row in valid.iterrows():
        usaf  = str(int(row['USAF'])).zfill(6) if pd.notna(row.get('USAF')) else '000000'
        wban  = str(int(row['WBAN'])).zfill(5) if pd.notna(row.get('WBAN')) else '99999'
        begin = str(int(row['BEGIN']))[:4]      if pd.notna(row.get('BEGIN')) else '?'
        end   = str(int(row['END']))[:4]        if pd.notna(row.get('END'))   else '?'
        alt   = round(float(row['ELEV(M)']), 1) if pd.notna(row.get('ELEV(M)')) else None

        result.append({
            'id':    f"{usaf}-{wban}",
            'name':  str(row.get('STATION NAME', '')),
            'lat':   round(float(row['LAT']), 5),
            'lon':   round(float(row['LON']), 5),
            'alt':   alt,
            'begin': begin,
            'end':   end,
        })
    return result


# ---------------------------------------------------------------------------
# MapDialog — full-screen selection dialog
# ---------------------------------------------------------------------------

class MapDialog(QDialog):
    """Full-screen modal dialog wrapping a MapWidget.

    Closes automatically when the user clicks "Select this station" in
    any popup.  Call :meth:`result_data` after ``exec()`` returns
    ``Accepted`` to retrieve the selection.

    Args:
        df_mf (pd.DataFrame, optional): Météo-France station catalogue.
        df_isd (pd.DataFrame, optional): NOAA ISD station catalogue.
        mf_client: Initialised ``mf_client.Client`` (needed to resolve
            full station_info for MF selections).
        isd_client: Initialised ``isd_client.ISDClient`` (needed for ISD).
        parent: Qt parent widget.
    """

    def __init__(
        self,
        df_mf: pd.DataFrame = None,
        df_isd: pd.DataFrame = None,
        mf_client=None,
        isd_client=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select a station — interactive map")
        self.resize(1200, 750)

        self._mf_client  = mf_client
        self._isd_client = isd_client
        self._df_isd     = df_isd
        self._result     = None   # (station_id_or_series, source, station_info)

        self._build_ui(df_mf, df_isd)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, df_mf, df_isd) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Info banner
        self._status = QLabel(
            "Click a marker to see station details, then press "
            "<b>Select this station</b> in the popup."
        )
        self._status.setStyleSheet(
            "color: #8b949e; font-size: 11px; padding: 2px 4px;"
        )
        layout.addWidget(self._status)

        # Map widget
        self._map = MapWidget(self)
        self._map.station_clicked.connect(self._on_map_click)
        layout.addWidget(self._map, stretch=1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        # Load tiles asynchronously — centre on France if only MF data,
        # otherwise world view
        if df_mf is not None and df_isd is None:
            self._map.load_stations(df_mf=df_mf, center_lat=46.5, center_lon=2.5, zoom=5)
        elif df_isd is not None and df_mf is None:
            self._map.load_stations(df_isd=df_isd)
        else:
            self._map.load_stations(df_mf=df_mf, df_isd=df_isd)

    # ------------------------------------------------------------------
    # Station selection handler
    # ------------------------------------------------------------------

    def _on_map_click(self, station_id: str, source: str) -> None:
        """Resolve a map click into a full station_info DataFrame.

        Args:
            station_id: Raw station ID from the map (MF ID or ISD
                ``USAF-WBAN`` composite).
            source: ``'mf'`` or ``'isd'``.
        """
        self._status.setText(f"Resolving station {station_id} ({source})…")

        try:
            if source == 'mf':
                station, station_info = self._resolve_mf(station_id)
            else:
                station, station_info = self._resolve_isd(station_id)
        except Exception as exc:
            logger.error("Could not resolve station %s: %s", station_id, exc)
            self._status.setText(f"Error: {exc}")
            return

        self._result = (station, source, station_info)
        row = station_info.iloc[0]
        self._status.setText(
            f"Selected: {row.get('Nom', '—')}  ({row.get('ID', '—')})"
        )
        logger.info(
            "Map station confirmed — %s (%s)", row.get('Nom', '?'), source
        )
        self.accept()

    def _resolve_mf(self, station_id: str):
        """Return (station_id_str, station_info_df) for an MF station.

        Args:
            station_id: DPClim station ID string.

        Returns:
            tuple: (str, pd.DataFrame)

        Raises:
            RuntimeError: If no MF client is available or the API call fails.
        """
        if self._mf_client is None:
            raise RuntimeError(
                "No Météo-France API client available. "
                "Configure your credentials first."
            )
        info = self._mf_client.get_station_info(station_id)
        if info is None or info.empty:
            raise RuntimeError(f"Station '{station_id}' not found via DPClim API.")
        return station_id, info

    def _resolve_isd(self, station_id: str):
        """Return (station_series, station_info_df) for an ISD station.

        Args:
            station_id: Composite ``USAF-WBAN`` key.

        Returns:
            tuple: (pd.Series, pd.DataFrame)

        Raises:
            RuntimeError: If the station cannot be found in the catalogue.
        """
        if self._isd_client is None:
            from src.isd_client import ISDClient
            self._isd_client = ISDClient()

        if self._df_isd is None:
            self._df_isd = self._isd_client._load_history()

        # Reconstruct USAF / WBAN from composite ID
        parts = station_id.split('-')
        if len(parts) != 2:
            raise RuntimeError(f"Invalid ISD station ID format: '{station_id}'.")
        usaf_str, wban_str = parts

        df = self._df_isd.copy()
        df['_usaf_str'] = df['USAF'].apply(
            lambda v: str(int(v)).zfill(6) if pd.notna(v) else '000000'
        )
        df['_wban_str'] = df['WBAN'].apply(
            lambda v: str(int(v)).zfill(5) if pd.notna(v) else '99999'
        )
        mask = (df['_usaf_str'] == usaf_str) & (df['_wban_str'] == wban_str)
        hits = df[mask]

        if hits.empty:
            raise RuntimeError(f"ISD station '{station_id}' not found in catalogue.")

        row  = hits.iloc[0]
        info = self._isd_client.get_station_info(row)
        return row, info

    # ------------------------------------------------------------------
    # Public result accessor
    # ------------------------------------------------------------------

    def result_data(self):
        """Return the confirmed selection after ``exec()`` returns Accepted.

        Returns:
            tuple: ``(station, source, station_info)`` where *station* is
                the raw station value (str for MF, pd.Series for ISD),
                *source* is ``'mf'`` or ``'isd'``, and *station_info* is
                a one-row DataFrame.  Returns ``None`` if no selection
                was made.
        """
        return self._result
