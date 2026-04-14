"""
Entry point — orchestrates the full data acquisition and reporting workflow.

Prompts the user to choose a data source (Météo-France DPClim or NOAA
ISD-Lite), guides station selection, downloads data, builds a
DatasetManager, saves the processed CSV, and generates quality-control
and climatological factsheet PDFs.

Author:  Samy KRAIEM
Created: 2024
Updated: 2026
"""

import logging
import unicodedata
import pandas as pd
from datetime import datetime
import configparser
import os
from src.mf_client import Client
from src.data_handler import DatasetManager
from src.isd_client import ISDClient
from src.data_quality_report import generate_quality_pdf
from src.data_facts_report import generate_factsheet_pdf
from src.utils import parse_coord
from src.station_selector import (load_mf_stations, select_mf_by_name,
                                   select_mf_nearest, select_mf_from_dept,
                                   select_isd_station)


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


# ---------------------------------------------------------------------------
# Source-specific acquisition flows
# ---------------------------------------------------------------------------

def _run_meteofrance():
    """Run the interactive Météo-France DPClim acquisition flow.

    Returns:
        tuple: (final_dataset, station_info, B_annee, E_annee, downloaded)
            where *downloaded* is the list of order IDs used for
            temporary-file cleanup.  All values are None on cancellation
            or unrecoverable error.
    """
    config = configparser.ConfigParser()
    config.read("API_config.txt")
    application_id = config.get("Parameters", "APPLICATION_ID")
    BASE_API_URL   = config.get("Parameters", "DATA_SERVER")
    api_key        = config.get("Parameters", "TOKEN").replace('"', '')

    logger.info("Credentials — APPLICATION_ID: %s | SERVER: %s | TOKEN: %s....",
                application_id, BASE_API_URL, api_key[:32])

    client = Client(api_key=api_key, base_url=BASE_API_URL)

    # --- Station selection mode ---
    print("\n  Station selection mode:")
    print("    [1]  By department")
    print("    [2]  By station ID  (direct entry)")
    print("    [3]  By name        (requires mf_stations.csv)")
    print("    [4]  By coordinates (requires mf_stations.csv)")
    while (sel_mode := input("\n  Mode: ").strip()) not in ('1', '2', '3', '4'):
        pass

    if sel_mode == '1':
        VALID_DEPTS = [str(i) for i in range(1, 96)] + [
            '971', '972', '973', '974', '975', '976', '977',
            '978', '984', '986', '987', '988', '989',
        ]
        while (DEPARTEMENT := input(
            "\nDepartment number (1–95, or 971/972/973/974/975/976/984/986/987/988/989): "
        ).strip()) not in VALID_DEPTS:
            pass
        stations = client.get_stations_list(DEPARTEMENT)
        if not stations:
            logger.error("Failed to retrieve station list for department %s.", DEPARTEMENT)
            return None, None, None, None, None
        selected_station = select_mf_from_dept(stations)

    elif sel_mode == '2':
        selected_station = input("\nEnter station ID: ").strip()

    elif sel_mode == '3':
        df_mf = load_mf_stations()
        selected_station = select_mf_by_name(df_mf)

    else:  # sel_mode == '4'
        df_mf = load_mf_stations()
        selected_station = select_mf_nearest(df_mf)

    station_info = client.get_station_info(selected_station)
    if station_info is None or station_info.empty:
        logger.error("Could not retrieve info for station '%s'. Check the ID and try again.",
                     selected_station)
        return None, None, None, None, None
    logger.info("Selected station:")
    for col in station_info.columns:
        logger.info("  %s : %s", col, station_info.iloc[0][col])

    annee_minimale = pd.to_datetime(station_info.iloc[0]['DateDebut']).year
    annee_maximale = (
        pd.to_datetime(station_info.iloc[0]['DateFin'], errors='coerce').year
        if not pd.isna(pd.to_datetime(station_info.iloc[0]['DateFin'], errors='coerce'))
        else datetime.now().year
    )

    while not (annee_minimale <= (B_annee := int(
        input("Please enter a starting year >= DateDebut (YYYY format) : ")
    )) <= annee_maximale):
        pass
    while not (B_annee <= (E_annee := int(
        input("Please enter a ending year <= DateFin (YYYY format)    : ")
    )) <= annee_maximale):
        pass

    logger.info("Order summary — station: %s | period: %d–%d",
                station_info.iloc[0]['ID'], B_annee, E_annee)
    if input("\nPlace Order ? (Y/n) : ").strip().lower() == 'n':
        logger.info("Order cancelled.")
        return None, None, None, None, None

    order_id = client.order_station_data(selected_station, B_annee, E_annee)
    if not order_id:
        return None, None, None, None, None
    logger.info("Order(s) placed successfully. IDs: %s", order_id)
    client.download_command_file(order_id)

    downloaded = [oid for oid in order_id if os.path.exists(f"command_{oid}_RAW_DATA.csv")]
    missing    = [oid for oid in order_id if oid not in downloaded]
    if missing:
        logger.warning("%d file(s) could not be downloaded and will be skipped: %s",
                       len(missing), missing)
    if not downloaded:
        logger.error("No files were downloaded. Aborting data processing.")
        return None, None, None, None, None

    dataset      = DatasetManager.from_csv(downloaded)
    final_dataset = dataset.create_subset()
    return final_dataset, station_info, B_annee, E_annee, downloaded


def _run_isd():
    """Run the interactive NOAA ISD-Lite acquisition flow.

    Returns:
        tuple: (final_dataset, station_info, B_annee, E_annee, None).
            The last element is always None because ISD data is kept in
            memory and no temporary files need to be cleaned up.  All
            values are None on cancellation or unrecoverable error.
    """
    isd = ISDClient()

    # Search mode selection
    print("\n  Station search mode:")
    print("    [1]  By name")
    print("    [2]  By coordinates (nearest station)")
    while (search_mode := input("\n  Mode: ").strip()) not in ('1', '2'):
        pass

    if search_mode == '1':
        # Name search — retry until at least one result
        while True:
            query    = input("\nSearch ISD station name (e.g. 'Kennedy', 'Paris', 'Toulouse'): ").strip()
            stations = isd.search_stations(query)
            if stations.empty:
                logger.warning("No station matches '%s'. Try another name.", query)
                continue
            break
    else:
        # Coordinate search
        print("  (accepted formats: 48.85  /  48,85  /  44,38°N  /  4,64°E)")
        while True:
            try:
                lat = parse_coord(input("\nLatitude  : "))
                lon = parse_coord(input("Longitude : "))
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    print("  Out-of-range coordinates. Please retry.")
                    continue
                break
            except ValueError as exc:
                print(f"  {exc}. Please retry.")
        n_nearest = 10
        logger.info("Searching %d nearest stations to (%.4f, %.4f) …", n_nearest, lat, lon)
        stations = isd.find_nearest_stations(lat, lon, n=n_nearest)

    selected_station = select_isd_station(stations)
    station_info     = isd.get_station_info(selected_station)
    logger.info("Selected station:")
    for col in station_info.columns:
        logger.info("  %s : %s", col, station_info.iloc[0][col])

    # Year bounds from isd-history BEGIN / END fields
    try:
        annee_minimale = int(str(int(selected_station['BEGIN']))[:4])
        annee_maximale = int(str(int(selected_station['END']))[:4])
    except (ValueError, TypeError):
        annee_minimale = 1900
        annee_maximale = datetime.now().year

    while not (annee_minimale <= (B_annee := int(
        input(f"Please enter a starting year >= {annee_minimale} (YYYY format) : ")
    )) <= annee_maximale):
        pass
    while not (B_annee <= (E_annee := int(
        input(f"Please enter an ending year  <= {annee_maximale} (YYYY format) : ")
    )) <= annee_maximale):
        pass

    logger.info("Order summary — station: %s | period: %d–%d",
                station_info.iloc[0]['ID'], B_annee, E_annee)
    if input("\nDownload data? (Y/n) : ").strip().lower() == 'n':
        logger.info("Download cancelled.")
        return None, None, None, None, None

    final_dataset = isd.download_and_parse(selected_station, B_annee, E_annee)
    return final_dataset, station_info, B_annee, E_annee, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():

    # --- Source selection ---------------------------------------------------
    print("\n  Select data source:")
    print("    [1]  Météo-France  (DPClim API — French stations)")
    print("    [2]  NOAA ISD-Lite (Integrated Surface Database — worldwide)")
    while (src := input("\n  Source: ").strip()) not in ('1', '2'):
        pass

    # --- Acquisition --------------------------------------------------------
    if src == '1':
        final_dataset, station_info, B_annee, E_annee, downloaded = _run_meteofrance()
    else:
        final_dataset, station_info, B_annee, E_annee, downloaded = _run_isd()

    if final_dataset is None:
        return

    # --- Quality overview ---------------------------------------------------
    quality_check = final_dataset.check_quality()

    logger.info("Missing data per parameter (%% missing):\n%s",
                quality_check.to_string())

    # --- Output directory and file names ------------------------------------
    nom_normalise = unicodedata.normalize('NFD', str(station_info.iloc[0]['Nom']))
    nom_normalise = (
        ''.join(c for c in nom_normalise if unicodedata.category(c) != 'Mn')
        .upper()
        .replace(' ', '-')
    )
    output_dir = f"out_{str(station_info.iloc[0]['ID'])}_{nom_normalise}"

    # --- Reports (QC first — flags are attached before CSV is written) ------
    all_checks = generate_quality_pdf(final_dataset, station_info, output_dir=output_dir)
    final_dataset.add_quality_flags(all_checks)

    final_dataset.save_subset_as_csv(
        station_name=str(station_info.iloc[0]['ID']) + '_' + nom_normalise,
        start_year=B_annee,
        end_year=E_annee,
        station_info=station_info,
        output_dir=output_dir,
    )

    generate_factsheet_pdf(final_dataset, station_info, output_dir=output_dir)

    # --- Cleanup (Météo-France temp files only) -----------------------------
    if downloaded:
        DatasetManager.delete_temporary_csvs(downloaded)


if __name__ == '__main__':
    main()
