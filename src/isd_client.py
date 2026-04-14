# -*- coding: utf-8 -*-
"""
ISD Client — NOAA Integrated Surface Database (ISD-Lite)

Downloads hourly meteorological data from NCEI and builds a DatasetManager
whose schema is compatible with the quality_report and factsheet pipeline.

ISD-Lite column layout (fixed-width, space-separated):
  1  Year                  4-digit
  2  Month                 01-12
  3  Day                   01-31
  4  Hour (UTC)            00-23
  5  Air temperature       °C x 10   (-9999 = missing)
  6  Dew-point temperature °C x 10   (-9999 = missing)
  7  Sea-level pressure    hPa x 10  (-9999 = missing)
  8  Wind direction        degrees   (-9999 = missing, 0 = calm)
  9  Wind speed            m/s x 10  (-9999 = missing)
 10  Total sky cover       oktas 0-8 (-9999 = missing, 9 = sky obscured)
 11  Liquid precip. 1 h    mm x 10   (-9999 = missing)
 12  Liquid precip. 6 h    mm x 10   (-9999 = missing)

Copyright 2024 Samy Kraiem — Apache License 2.0
"""

import gzip
import io
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from src.data_handler import DatasetManager

logger = logging.getLogger(__name__)

_ISD_STATIONS_CSV = os.path.join("weather_stations_infos", "isd_stations.csv")
_ISD_LITE_BASE    = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
_MISSING_FLAG    = -9999
_MAX_RESULTS     = 30
_REQUEST_PAUSE   = 0.4   # seconds between yearly downloads (gentle rate-limit)


class ISDClient:
    """Client for NOAA ISD-Lite data (HTTPS, no FTP, no tkinter)."""

    def __init__(self):
        self._history: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Station catalogue
    # ------------------------------------------------------------------

    def _load_history(self) -> pd.DataFrame:
        """
        Load the ISD station catalogue from the local CSV.
        Exits with a clear message if the file is not found — run
        fetch_isd_stations.py first to generate it.
        """
        if self._history is not None:
            return self._history
        if not os.path.exists(_ISD_STATIONS_CSV):
            logger.error(
                "ISD station catalogue '%s' not found. "
                "Run fetch_isd_stations.py first to generate it.",
                _ISD_STATIONS_CSV,
            )
            sys.exit(1)
        df = pd.read_csv(_ISD_STATIONS_CSV, dtype={'USAF': str, 'WBAN': str})
        for col in ('LAT', 'LON', 'ELEV(M)'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ('USAF', 'WBAN', 'BEGIN', 'END'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        self._history = df
        logger.info("ISD catalogue loaded — %d stations.", len(df))
        return df

    def search_stations(self, query: str) -> pd.DataFrame:
        """
        Return stations whose name contains *query* (case-insensitive).
        Results capped at _MAX_RESULTS to avoid unmanageable lists.
        """
        df = self._load_history()
        mask = df['STATION NAME'].str.contains(query, case=False, na=False)
        hits = df[mask].reset_index(drop=True)
        if len(hits) > _MAX_RESULTS:
            logger.warning(
                "%d stations match '%s' — showing first %d. Refine your query.",
                len(hits), query, _MAX_RESULTS,
            )
            hits = hits.iloc[:_MAX_RESULTS].reset_index(drop=True)
        return hits

    def find_nearest_stations(
        self, lat: float, lon: float, n: int = 10
    ) -> pd.DataFrame:
        """
        Return the *n* stations closest to (lat, lon) sorted by distance.
        Distance is computed with the Haversine formula (great-circle, km).
        A 'distance_km' column is appended to the result.
        """
        df = self._load_history().dropna(subset=['LAT', 'LON'])

        # Vectorised Haversine
        lat_r  = math.radians(lat)
        lon_r  = math.radians(lon)
        dlat   = np.radians(df['LAT'].values) - lat_r
        dlon   = np.radians(df['LON'].values) - lon_r
        a      = (np.sin(dlat / 2) ** 2
                  + math.cos(lat_r) * np.cos(np.radians(df['LAT'].values))
                  * np.sin(dlon / 2) ** 2)
        dist   = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

        df = df.copy()
        df['distance_km'] = dist
        return (
            df.nsmallest(n, 'distance_km')
            .reset_index(drop=True)
        )

    def get_station_info(self, station: pd.Series) -> pd.DataFrame:
        """
        Build a station_info DataFrame with the same column schema used by
        the MeteoFrance pipeline (ID, Nom, Altitude, Latitude, Longitude,
        DateDebut, DateFin).
        """
        usaf = str(int(station['USAF'])).zfill(6)
        wban = str(int(station['WBAN'])).zfill(5) if pd.notna(station['WBAN']) else '99999'

        def _fmt(raw_int):
            s = str(int(raw_int)) if pd.notna(raw_int) else '00000000'
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else s

        return pd.DataFrame([{
            'ID':        f"{usaf}-{wban}",
            'Nom':       str(station['STATION NAME']),
            'Altitude':  station['ELEV(M)'],
            'Latitude':  station['LAT'],
            'Longitude': station['LON'],
            'DateDebut': _fmt(station.get('BEGIN')),
            'DateFin':   _fmt(station.get('END')),
        }])

    # ------------------------------------------------------------------
    # Data download and parsing
    # ------------------------------------------------------------------

    def download_and_parse(
        self,
        station: pd.Series,
        start_year: int,
        end_year: int,
    ) -> DatasetManager:
        """
        Download ISD-Lite files year by year and return a DatasetManager
        compatible with quality_report and factsheet.

        Columns produced:
          DATE, T, TD, U, UABS, PSTAT (= SLP), DD, FF, N, RR1

        U and UABS are derived from T and TD via the August-Roche-Magnus
        formula (no ISD field; computed on the fly).
        GLO / DIR / DIF / INFRAR are absent — factsheet will skip radiation.

        Note: PSTAT contains sea-level pressure (SLP), the only pressure
        variable available in ISD-Lite.
        """
        usaf = str(int(station['USAF'])).zfill(6)
        wban = str(int(station['WBAN'])).zfill(5) if pd.notna(station['WBAN']) else '99999'

        frames = []
        for year in range(start_year, end_year + 1):
            df_year = self._download_year(usaf, wban, year)
            if df_year is not None:
                frames.append(df_year)
            time.sleep(_REQUEST_PAUSE)

        if not frames:
            raise RuntimeError(
                "No ISD-Lite data could be retrieved for the requested period."
            )

        raw  = pd.concat(frames, ignore_index=True)
        data = self._parse_raw(raw)
        logger.info(
            "ISD dataset built — %d hourly records (%d–%d).",
            len(data), start_year, end_year,
        )
        return DatasetManager(data)

    def _download_year(self, usaf: str, wban: str, year: int) -> pd.DataFrame | None:
        """Download and decompress one ISD-Lite year file. Returns None on failure."""
        url = f"{_ISD_LITE_BASE}/{year}/{usaf}-{wban}-{year}.gz"
        for attempt in range(1, 4):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    logger.warning("Year %d — no file on server (404). Skipping.", year)
                    return None
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                logger.warning("Year %d attempt %d/3 failed: %s", year, attempt, exc)
                if attempt == 3:
                    logger.error("Year %d skipped after 3 failures.", year)
                    return None
                time.sleep(10)

        logger.info("  ↳ %d  (%d kB)", year, len(r.content) // 1024)
        rows = []
        with gzip.open(io.BytesIO(r.content)) as gz:
            for line in gz.read().decode('latin-1').splitlines():
                parts = line.split()
                if len(parts) >= 12:
                    rows.append(parts[:12])

        if not rows:
            return None

        return pd.DataFrame(rows, columns=[
            '_yr', '_mo', '_da', '_hr',
            '_T', '_TD', '_SLP', '_DD', '_FF', '_N', '_RR1', '_RR6',
        ]).astype(int)

    @staticmethod
    def _parse_raw(raw: pd.DataFrame) -> pd.DataFrame:
        """Scale integer ISD-Lite columns to physical units; -9999 → NaN."""

        def _scale(col: str, factor: float = 10.0) -> pd.Series:
            s = raw[col].astype(float)
            s[s == _MISSING_FLAG] = np.nan
            return s / factor

        def _flag(col: str) -> pd.Series:
            s = raw[col].astype(float)
            s[s == _MISSING_FLAG] = np.nan
            return s

        dates = pd.to_datetime(
            raw['_yr'].astype(str).str.zfill(4) +
            raw['_mo'].astype(str).str.zfill(2) +
            raw['_da'].astype(str).str.zfill(2) +
            raw['_hr'].astype(str).str.zfill(2),
            format='%Y%m%d%H',
        )

        T  = _scale('_T')
        TD = _scale('_TD')

        # Cloud cover: oktas 0–8; value 9 = sky obscured → NaN
        N = _flag('_N')
        N[N >= 9] = np.nan

        # Relative humidity and absolute humidity from T and dew-point
        # using the August-Roche-Magnus approximation
        with np.errstate(invalid='ignore'):
            e    = 6.112 * np.exp(17.67 * TD / (TD + 243.5))   # actual vapour pressure (hPa)
            es   = 6.112 * np.exp(17.67 * T  / (T  + 243.5))   # saturation vapour pressure (hPa)
            U    = (100.0 * e / es).clip(0, 100)                 # relative humidity (%)
            UABS = (216.7 * e / (T + 273.15)).clip(0)            # absolute humidity (g/m³)

        return pd.DataFrame({
            'DATE':  dates,
            'T':     T,
            'TD':    TD,
            'U':     U,
            'UABS':  UABS,
            'PSTAT': _scale('_SLP'),  # sea-level pressure — best available in ISD-Lite
            'DD':    _flag('_DD'),    # degrees (0 = calm, 360 = true North)
            'FF':    _scale('_FF'),   # m/s
            'N':     N,               # oktas 0–8
            'RR1':   _scale('_RR1'),  # mm
        })
