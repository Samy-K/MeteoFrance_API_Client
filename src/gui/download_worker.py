"""
Background worker thread for the full data acquisition pipeline.

Runs Météo-France DPClim or NOAA ISD-Lite downloads entirely off the
main thread so the PyQt6 GUI stays responsive.  Progress is communicated
back to the main window via Qt signals.

Author:  Samy KRAIEM
Created: 2026
Updated: 2026
"""

import logging
import os
import unicodedata

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe to use in a worker thread

import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from src.data_handler import DatasetManager
from src.data_facts_report import generate_factsheet_pdf
from src.data_quality_report import generate_quality_pdf
from src.isd_client import ISDClient
from src.mf_client import Client

logger = logging.getLogger(__name__)


class DownloadWorker(QThread):
    """QThread that runs the full acquisition + report pipeline.

    Signals:
        finished (DatasetManager, pd.DataFrame, int, int, str):
            Emitted on success with (dataset, station_info, start_year,
            end_year, output_dir).
        error (str):
            Emitted when an unrecoverable error occurs.
    """

    finished = pyqtSignal(object, object, int, int, str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        source: str,
        station,
        station_info: pd.DataFrame,
        start_year: int,
        end_year: int,
        api_config: dict = None,
        parent=None,
    ):
        """Initialise the worker.

        Args:
            source (str): ``'mf'`` or ``'isd'``.
            station: Station ID string (MF) or pd.Series row (ISD).
            station_info (pd.DataFrame): One-row station metadata DataFrame.
            start_year (int): First year to download (inclusive).
            end_year (int): Last year to download (inclusive).
            api_config (dict, optional): Credentials for MF — keys
                ``APPLICATION_ID``, ``TOKEN``, ``DATA_SERVER``.
            parent: Qt parent object.
        """
        super().__init__(parent)
        self.source       = source
        self.station      = station
        self.station_info = station_info
        self.start_year   = start_year
        self.end_year     = end_year
        self.api_config   = api_config or {}

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        """Execute the acquisition pipeline (called automatically by Qt)."""
        try:
            if self.source == 'mf':
                self._run_mf()
            else:
                self._run_isd()
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalised_name(self) -> str:
        """Return an ASCII-safe, upper-case version of the station name."""
        nom = unicodedata.normalize('NFD', str(self.station_info.iloc[0]['Nom']))
        return (
            ''.join(c for c in nom if unicodedata.category(c) != 'Mn')
            .upper()
            .replace(' ', '-')
        )

    def _output_dir(self) -> str:
        """Build the output directory name from station ID and name."""
        return f"out_{self.station_info.iloc[0]['ID']}_{self._normalised_name()}"

    def _generate_reports(
        self,
        dataset: DatasetManager,
        output_dir: str,
    ) -> None:
        """Run QC report → flag → save CSV → factsheet.

        Args:
            dataset (DatasetManager): Processed hourly dataset.
            output_dir (str): Target output directory.
        """
        station_name = f"{self.station_info.iloc[0]['ID']}_{self._normalised_name()}"

        all_checks = generate_quality_pdf(dataset, self.station_info, output_dir=output_dir)
        dataset.add_quality_flags(all_checks)
        dataset.save_subset_as_csv(
            station_name=station_name,
            start_year=self.start_year,
            end_year=self.end_year,
            station_info=self.station_info,
            output_dir=output_dir,
        )
        generate_factsheet_pdf(dataset, self.station_info, output_dir=output_dir)

    def _run_mf(self) -> None:
        """Météo-France DPClim acquisition pipeline."""
        client = Client(
            api_key=self.api_config.get('TOKEN') or None,
            application_id=self.api_config.get('APPLICATION_ID') or None,
            base_url=self.api_config.get('DATA_SERVER'),
        )

        order_ids = client.order_station_data(
            self.station, self.start_year, self.end_year
        )
        if not order_ids:
            self.error.emit("No orders could be placed. Check your API credentials.")
            return

        client.download_command_file(order_ids)

        downloaded = [
            oid for oid in order_ids
            if os.path.exists(f"command_{oid}_RAW_DATA.csv")
        ]
        missing = [oid for oid in order_ids if oid not in downloaded]
        if missing:
            logger.warning("%d file(s) skipped (download failed): %s", len(missing), missing)
        if not downloaded:
            self.error.emit("No CSV files were downloaded.")
            return

        dataset    = DatasetManager.from_csv(downloaded).create_subset()
        output_dir = self._output_dir()
        self._generate_reports(dataset, output_dir)
        DatasetManager.delete_temporary_csvs(downloaded)

        self.finished.emit(dataset, self.station_info, self.start_year, self.end_year, output_dir)

    def _run_isd(self) -> None:
        """NOAA ISD-Lite acquisition pipeline."""
        isd     = ISDClient()
        dataset = isd.download_and_parse(self.station, self.start_year, self.end_year)

        output_dir = self._output_dir()
        self._generate_reports(dataset, output_dir)

        self.finished.emit(dataset, self.station_info, self.start_year, self.end_year, output_dir)
