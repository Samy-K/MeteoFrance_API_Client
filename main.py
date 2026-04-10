# -*- coding: utf-8 -*-
"""
Created onWed Jan 17 18:07:13 2024

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
import unicodedata
import pandas as pd
from datetime import datetime
import configparser
import os
from client import Client
from Data_handler import DatasetManager

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    '\033[36m',    # cyan
        logging.INFO:     '\033[32m',    # green
        logging.WARNING:  '\033[33m',    # yellow
        logging.ERROR:    '\033[31m',    # red
        logging.CRITICAL: '\033[1;31m',  # bold red
    }
    RESET = '\033[0m'
    WHITE = '\033[97m'

    def format(self, record):
        color   = self.COLORS.get(record.levelno, self.RESET)
        ts      = self.formatTime(record, self.datefmt)
        level   = f"{color}[{record.levelname}]{self.RESET}"
        message = f"{self.WHITE}{record.getMessage()}{self.RESET}"
        return f"{ts} {level} {message}"

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(datefmt='%Y-%m-%d %H:%M:%S'))
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(handler)

logger = logging.getLogger(__name__)

def main():

    config = configparser.ConfigParser()
    config.read("API_config.txt")

    application_id = config.get("Parameters", "APPLICATION_ID")
    BASE_API_URL   = config.get("Parameters", "DATA_SERVER")
    api_key        = config.get("Parameters", "TOKEN").replace('"', '')

    logger.info("Credentials — APPLICATION_ID: %s | SERVER: %s | TOKEN: %s....",
                application_id, BASE_API_URL, api_key[:32])

    client = Client(api_key=api_key, base_url=BASE_API_URL)

    # Department choice
    while (DEPARTEMENT := input("\nEnter the desired department number. Numbers range from 1 to 95, plus 971, 972, 973, 974, 975, 984, 985, 986, 987, 988 for French overseas departments and territories.\n\nDepartment :"\
                                ).strip()) not in [str(i) for i in range(1, 96)] + ['971', '972', '973', '974', '975', '976', '977', '978', '984', '986', '987', '988', '989']: pass

    # Get the department weather station list
    stations = client.get_stations_list(DEPARTEMENT)
    if stations:
        # Station choice
        selected_station = client.select_station(stations)

        # Retrieve selected station information
        station_info = client.get_station_info(selected_station)
        logger.info("Selected station:")
        for col in station_info.columns:
            logger.info("  %s : %s", col, station_info.iloc[0][col])

        # Time period selection
        annee_minimale = pd.to_datetime(station_info.iloc[0]['DateDebut']).year
        annee_maximale = pd.to_datetime(station_info.iloc[0]['DateFin'], errors='coerce').year if not \
            pd.isna(pd.to_datetime(station_info.iloc[0]['DateFin'], errors='coerce')) else datetime.now().year

        while not (annee_minimale <= (B_annee := int(input("Please enter a starting year >= DateDebut (YYYY format) : "))) <= annee_maximale): pass
        while not (B_annee        <= (E_annee := int(input("Please enter a ending year <= DateFin (YYYY format)    : "))) <= annee_maximale): pass

        # Place order
        logger.info("Order summary — station: %s | period: %d–%d",
                    station_info.iloc[0]['ID'], B_annee, E_annee)
        confirmation = 'Y' if input("\nPlace Order ? (Y/n) : ").strip().lower() != 'n' else 'n'
        order_id = []
        if confirmation == 'Y':
            order_id = client.order_station_data(selected_station, B_annee, E_annee)
            if order_id:
                logger.info("Order(s) placed successfully. IDs: %s", order_id)
                client.download_command_file(order_id)
        else:
            logger.info("Order cancelled.")
            return
    else:
        logger.error("Failed to retrieve station list for department %s.", DEPARTEMENT)
        return

    # Formatting data
    downloaded = [oid for oid in order_id if os.path.exists(f"command_{oid}_RAW_DATA.csv")]
    missing    = [oid for oid in order_id if oid not in downloaded]
    if missing:
        logger.warning("%d file(s) could not be downloaded and will be skipped: %s", len(missing), missing)
    if not downloaded:
        logger.error("No files were downloaded. Aborting data processing.")
        return

    dataset       = DatasetManager.from_csv(downloaded)
    final_dataset = dataset.create_subset()
    quality_check = final_dataset.check_quality()    # Check for missing data
    parameters    = final_dataset.list_parameters()  # List all parameters
    basic_stats   = final_dataset.basic_statistics() # Basic statistical insights
    data_summary  = final_dataset.data_summary()     # Summary of the dataset

    logger.info("Missing data per parameter (%%  missing):\n%s",
                quality_check.to_string())

    # Writing data
    nom_normalise = unicodedata.normalize('NFD', str(station_info.iloc[0]['Nom']))
    nom_normalise = ''.join(c for c in nom_normalise if unicodedata.category(c) != 'Mn').upper().replace(' ', '-')
    final_dataset.save_subset_as_csv(station_name = str(station_info.iloc[0]['ID']) + '_' + nom_normalise,
                                       start_year=B_annee, end_year=E_annee, station_info=station_info)
    # Cleaning
    DatasetManager.delete_temporary_csvs(downloaded)

if __name__ == '__main__':
    main()
