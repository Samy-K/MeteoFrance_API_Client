"""
NOAA ISD station catalogue downloader.

Downloads the NOAA ISD station catalogue (``isd-history.csv``) from NCEI
in a single HTTP request and saves it to
``weather_stations_infos/isd_stations.csv``.

Run this script once before using ISDClient, or re-run at any time to
refresh the catalogue.  Completes in a few seconds.

Author:  Samy KRAIEM
Created: 2024
Updated: 2026
"""

import io
import logging
import os

import pandas as pd
import requests

OUTPUT_DIR  = "weather_stations_infos"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "isd_stations.csv")
SOURCE_URL  = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Downloading ISD station catalogue from NCEI …")
    r = requests.get(SOURCE_URL, timeout=90)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]

    for col in ('LAT', 'LON', 'ELEV(M)'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ('USAF', 'WBAN', 'BEGIN', 'END'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # The NOAA source occasionally contains duplicate USAF+WBAN entries.
    # Keep the row with the most recent END date for each composite key.
    before = len(df)
    df = (df
          .sort_values('END', ascending=False, na_position='last')
          .drop_duplicates(subset=['USAF', 'WBAN'], keep='first')
          .reset_index(drop=True))
    dropped = before - len(df)
    if dropped:
        logger.info("Removed %d duplicate USAF+WBAN row(s) from source data.", dropped)

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    logger.info("Saved %d stations -> %s", len(df), OUTPUT_CSV)


if __name__ == '__main__':
    main()
