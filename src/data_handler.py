"""
Dataset manager for post-download meteorological data processing.

Reads, subsets, quality-flags, and saves hourly weather data from both
Météo-France DPClim (CSV order files) and NOAA ISD-Lite (in-memory
DataFrames).  Acts as the central data container passed through the
quality-report and factsheet pipeline.

Author:  Samy KRAIEM
Created: 2024
Updated: 2026
"""

import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, data: pd.DataFrame) -> None:
        """Wrap a pandas DataFrame for use in the quality-report pipeline.

        Args:
            data (pd.DataFrame): Hourly meteorological data.  Must contain
                at least a ``DATE`` column (datetime64) once ``create_subset``
                has been called.
        """
        self.data = data

    @classmethod
    def from_csv(cls, order_ids: list, separator: str = ';',
                 decimal: str = ',') -> 'DatasetManager':
        """Build a DatasetManager by reading and concatenating per-order CSV files.

        Each Météo-France order is downloaded as an individual CSV named
        ``command_{order_id}_RAW_DATA.csv``.  This method reads all files,
        concatenates them row-wise, and returns the combined dataset.

        Args:
            order_ids (list): Order ID strings returned by
                ``Client.order_station_data()``.
            separator (str): CSV column separator (default ``';'``).
            decimal (str): Decimal separator used in the CSV (default ``','``).

        Returns:
            DatasetManager: Instance wrapping the concatenated raw data.
        """
        all_data_frames = []

        for order_id in order_ids:
            file_path = f'command_{order_id}_RAW_DATA.csv'
            data = pd.read_csv(file_path, sep=separator, decimal=decimal, encoding='utf-8')
            all_data_frames.append(data)

        concatenated_data = pd.concat(all_data_frames, ignore_index=True)
        return cls(concatenated_data)

    def check_quality(self) -> pd.Series:
        """Return the percentage of missing values for each column.

        Returns:
            pd.Series: Index = column names, values = percentage missing (0–100).
        """
        total_rows         = self.data.shape[0]
        missing_data       = self.data.isnull().sum()
        missing_percentage = (missing_data / total_rows) * 100
        return missing_percentage

    def list_parameters(self) -> list:
        """Return the list of column names present in the dataset.

        Returns:
            list: Column names as strings.
        """
        return self.data.columns.tolist()

    def basic_statistics(self) -> pd.DataFrame:
        """Return descriptive statistics for all numeric columns.

        Returns:
            pd.DataFrame: Output of ``pandas.DataFrame.describe()``.
        """
        return self.data.describe()

    def data_summary(self) -> dict:
        """Return a brief summary of the dataset.

        Returns:
            dict: Keys ``'Data Types'`` (dtypes Series) and
                ``'First Five Rows'`` (first 5-row DataFrame).
        """
        return {
            "Data Types": self.data.dtypes,
            "First Five Rows": self.data.head()
        }

    def create_subset(self, columns: list = None) -> 'DatasetManager':
        """Extract a standard meteorological subset and parse the DATE column.

        Selects a predefined set of columns (or a caller-supplied list),
        warns about any that are absent, and parses DATE as ``%Y%m%d%H``.

        Args:
            columns (list, optional): Column names to keep.  Defaults to
                ``['DATE', 'PSTAT', 'T', 'UABS', 'U', 'TD', 'GLO', 'DIR',
                'DIF', 'N', 'INFRAR', 'DD', 'FF', 'RR1']``.

        Returns:
            DatasetManager: New instance containing only the selected columns
                with DATE parsed as datetime64.

        Raises:
            ValueError: If none of the requested columns are present.
        """
        default_columns = ['DATE', 'PSTAT', 'T', 'UABS', 'U', 'TD', 'GLO', 'DIR', 'DIF', 'N', 'INFRAR', 'DD', 'FF', 'RR1']
        if columns is None:
            columns = default_columns
        missing = [c for c in columns if c not in self.data.columns]
        if missing:
            logger.warning("Column(s) absent from dataset and skipped: %s", missing)
            columns = [c for c in columns if c in self.data.columns]
        if not columns:
            raise ValueError("No requested columns are present in the dataset.")
        subset_data = self.data[columns].copy()
        subset_data['DATE'] = pd.to_datetime(subset_data['DATE'], format='%Y%m%d%H')
        return DatasetManager(subset_data)

    def save_subset_as_csv(self, station_name: str, start_year: int,
                           end_year: int, station_info: pd.DataFrame,
                           output_dir: str = "out") -> None:
        """Write the dataset to a commented CSV file in *output_dir*.

        Station metadata is prepended as ``#``-prefixed comment lines so
        the file is self-describing.

        Output path: ``{output_dir}/RAW_DATA_{station_name}_{start_year}-{end_year}.csv``

        Args:
            station_name (str): Station identifier used in the file name
                (e.g. ``'84064001_LES-ILES'``).
            start_year (int): First year of the requested period.
            end_year (int): Last year of the requested period.
            station_info (pd.DataFrame): One-row DataFrame with station
                metadata columns (ID, Nom, Altitude, …).
            output_dir (str): Directory in which to write the output CSV.
                Created if it does not exist.
        """
        file_name = os.path.join(output_dir, "RAW_DATA_" + station_name + '_' + str(start_year) + '-' + str(end_year) + '.csv')

        _col_en = {
            'ID': 'ID', 'Nom': 'Name', 'Altitude': 'Altitude',
            'Latitude': 'Latitude', 'Longitude': 'Longitude',
            'DateDebut': 'StartDate', 'DateFin': 'EndDate',
        }
        with open(file_name, 'w') as fichier:
            for col in station_info.columns:
                label = _col_en.get(col, col)
                fichier.write(f"#{label:10} : {station_info.iloc[0][col]}\n")

        self.data.to_csv(file_name, index=False, mode='a')

        logger.info("Subset saved as %s", file_name)

    def add_quality_flags(self, all_checks: dict) -> None:
        """Attach per-variable quality-flag columns to the internal DataFrame.

        For each variable in *all_checks* a column named ``FLAG_{var}`` is
        added in place.  An empty string means the observation is valid;
        a non-empty string contains one or more pipe-separated flag codes
        (e.g. ``'MISSING'`` or ``'ABSURD|CONTEXTUAL'``).

        Must be called **after** ``generate_quality_pdf()`` and **before**
        ``save_subset_as_csv()`` so that flags are included in the output CSV.

        Args:
            all_checks (dict): Mapping ``{var: check_dict}`` as returned by
                ``generate_quality_pdf()``.  Each ``check_dict`` must contain
                a ``'flags'`` Series aligned with ``self.data``.
        """
        for var, chk in all_checks.items():
            self.data[f'FLAG_{var}'] = chk['flags'].values

    @staticmethod
    def delete_temporary_csvs(order_ids: list) -> None:
        """Delete the temporary per-order CSV files written during download.

        Args:
            order_ids (list): Order ID strings whose corresponding files
                (``command_{id}_RAW_DATA.csv``) should be removed.
                Files that do not exist on disk are silently skipped.
        """
        for order_id in order_ids:
            file_path = f'command_{order_id}_RAW_DATA.csv'
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Deleted temporary file: %s", file_path)
