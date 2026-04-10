# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:07:58 2024

@author: Samy-K

Copyright 2024 Samy Kraiem

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, data):
        """
        Initialize the DatasetManager with a pandas DataFrame.
        """
        self.data = data

    @classmethod
    def from_csv(cls, order_ids, separator=';', decimal=','):
        """
        Class method to create an instance of DatasetManager from multiple CSV files.
        Each CSV file corresponds to an order_id and they are concatenated into one DataFrame.
        """
        all_data_frames = []

        for order_id in order_ids:
            file_path = f'command_{order_id}_RAW_DATA.csv'
            data = pd.read_csv(file_path, sep=separator, decimal=decimal, encoding='utf-8')
            all_data_frames.append(data)

        concatenated_data = pd.concat(all_data_frames, ignore_index=True)
        return cls(concatenated_data)

    def check_quality(self):
        """
        Check for missing or inconsistent data and calculate the percentage of missing values per column.
        """
        total_rows         = self.data.shape[0]
        missing_data       = self.data.isnull().sum()
        missing_percentage = (missing_data / total_rows) * 100
        return missing_percentage

    def list_parameters(self):
        """
        List all the parameters (columns) in the dataset.
        """
        return self.data.columns.tolist()

    def basic_statistics(self):
        """
        Provide basic statistical insights about the dataset.
        """
        return self.data.describe()

    def data_summary(self):
        """
        Display a summary of the dataset including types of data and sample values.
        """
        return {
            "Data Types": self.data.dtypes,
            "First Five Rows": self.data.head()
        }

    def create_subset(self, columns=None):
        """
        Creates a subset of the data based on the provided column names.
        The subset is returned as a new DatasetManager instance.
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

    def save_subset_as_csv(self, station_name, start_year, end_year, station_info):
        """
        Saves the subset as a CSV file, filename is composed with station name, start year, and end year.
        """
        file_name = "RAW_DATA_" + station_name + '_' + str(start_year) + '-' + str(end_year) + '.csv'

        with open(file_name, 'w') as fichier:
            for col in station_info.columns:
                fichier.write(f"#{col:10} : {station_info.iloc[0][col]}\n")

        self.data.to_csv(file_name, index=False, mode='a')

        logger.info("Subset saved as %s", file_name)

    @staticmethod
    def delete_temporary_csvs(order_ids):
        """
        Deletes all temporary CSV files corresponding to each order_id.
        """
        for order_id in order_ids:
            file_path = f'command_{order_id}_RAW_DATA.csv'
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Deleted temporary file: %s", file_path)
