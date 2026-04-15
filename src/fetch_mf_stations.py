"""
Météo-France DPClim station catalogue builder.

Queries the DPClim API for every French department, collects all unique
hourly station IDs, then fetches full metadata for each station and writes
the result to ``weather_stations_infos/mf_stations.csv`` (one row per
station, most-recent record of operation kept).

Strategy
--------
1. Query ``/liste-stations/horaire`` for every department — collect unique IDs.
2. Query ``/information-station`` for every ID — collect coordinates and dates.
3. Write ``mf_stations.csv``.

Rate limiting
-------------
Hard cap: 100 req/min.  This script targets ~85 req/min
(``REQUEST_INTERVAL = 0.7 s``).  On a 429 response the script backs off
exponentially before retrying.

Resume / checkpoint
-------------------
If ``mf_stations.csv`` already exists the script reads it, skips
already-fetched station IDs, and appends new rows.  Safe to interrupt and
restart at any point.

Expected runtime: ~25 minutes for ~2 000 stations (fresh run).

Author:  Samy KRAIEM
Created: 2024
Updated: 2026
"""

import configparser
import logging
import os
import sys
import time
from datetime import datetime
# Allow running directly from project root: python src/fetch_mf_stations.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.mf_client import Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEPARTMENTS = (
    [str(i) for i in range(1, 96)]
    + ['971', '972', '973', '974', '975', '976', '977',
       '978', '984', '986', '987', '988', '989']
)

OUTPUT_DIR        = "weather_stations_infos"
OUTPUT_CSV        = os.path.join(OUTPUT_DIR, "mf_stations.csv")
REQUEST_INTERVAL  = 0.7   # seconds between requests  (~85 req/min)
CHECKPOINT_EVERY  = 100   # save CSV every N stations
MAX_RETRIES       = 3     # retries on 429 or transient error

# ---------------------------------------------------------------------------
# Logging (plain format — no colour, designed for long batch runs)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_dept_stations(client: Client, dept: str) -> list:
    """
    Fetch the station list for one department with 429-aware retry.
    Bypasses client.get_stations_list() to get full access to the status code.
    Returns a list of station dicts, or [] on unrecoverable failure.
    """
    url = client.base_url + "/liste-stations/horaire?id-departement=" + str(dept)
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.request('GET', url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = 60 * attempt
            logger.warning(
                "Department %s — 429 rate-limited, attempt %d/%d, waiting %ds.",
                dept, attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        else:
            logger.warning(
                "Department %s — HTTP %d: %s",
                dept, response.status_code, response.text[:120],
            )
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL * 5)
            else:
                return []
    logger.error("Department %s — gave up after %d attempts.", dept, MAX_RETRIES)
    return []


def _fetch_station_info(client: Client, station_id: str) -> dict | None:
    """
    Call /information-station with 429-aware retry.
    Returns a flat dict (most-recent record) or None on failure.
    """
    url = client.base_url + "/information-station?id-station=" + station_id

    for attempt in range(1, MAX_RETRIES + 1):
        response = client.request('GET', url)

        if response.status_code == 200:
            data = response.json()
            if not data:
                return None
            # Keep the last item (most recent period of operation)
            item = data[-1]
            pos  = item.get('positions', [{}])[-1] if item.get('positions') else {}
            typ  = (item.get('typesPoste') or [{}])[-1].get('type')
            return {
                'ID':        item.get('id'),
                'Nom':       item.get('nom'),
                'LieuDit':   item.get('lieuDit'),
                'Bassin':    item.get('bassin'),
                'DateDebut': item.get('dateDebut'),
                'DateFin':   item.get('dateFin'),
                'Type':      typ,
                'Altitude':  pos.get('altitude'),
                'Latitude':  pos.get('latitude'),
                'Longitude': pos.get('longitude'),
            }

        elif response.status_code == 429:
            wait = REQUEST_INTERVAL + 60 * attempt   # 60s, 120s, 180s
            logger.warning(
                "Station %s — 429 rate-limited, attempt %d/%d, waiting %.0f s.",
                station_id, attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)

        else:
            logger.warning(
                "Station %s — HTTP %d: %s",
                station_id, response.status_code, response.text[:120],
            )
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL * 5)
            else:
                return None

    logger.error("Station %s — gave up after %d attempts.", station_id, MAX_RETRIES)
    return None


def _load_checkpoint() -> tuple[pd.DataFrame, set]:
    """Load existing CSV if present; return (df, set_of_already_fetched_ids)."""
    try:
        df = pd.read_csv(OUTPUT_CSV, dtype={'ID': str})
        done = set(df['ID'].dropna().astype(str))
        logger.info("Checkpoint loaded — %d stations already in %s.", len(done), OUTPUT_CSV)
        return df, done
    except FileNotFoundError:
        return pd.DataFrame(), set()


def _save(rows: list[dict], existing_df: pd.DataFrame) -> pd.DataFrame:
    """Append *rows* to *existing_df*, deduplicate by ID, write CSV.

    Deduplication keeps the last occurrence of each station ID so that a
    re-run or a resumed checkpoint never creates duplicate rows.
    """
    new_df   = pd.DataFrame(rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=['ID'], keep='last').reset_index(drop=True)
    dropped  = before - len(combined)
    if dropped:
        logger.warning("Removed %d duplicate station ID(s) before saving.", dropped)
    combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = datetime.now()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Mode selection -----------------------------------------------------
    csv_exists = os.path.exists(OUTPUT_CSV)
    if csv_exists:
        print(f"\n  '{OUTPUT_CSV}' already exists.")
        print("    [1]  Complete  — skip stations already present in the CSV")
        print("    [2]  Rebuild   — delete the CSV and start from scratch")
        while (run_mode := input("\n  Mode: ").strip()) not in ('1', '2'):
            pass
    else:
        run_mode = '2'   # no existing file → always full build

    if run_mode == '2' and csv_exists:
        os.remove(OUTPUT_CSV)
        logger.info("Existing CSV deleted — full rebuild starting.")

    # --- Credentials --------------------------------------------------------
    config = configparser.ConfigParser()
    config.read("API_config.txt")
    api_key  = config.get("Parameters", "TOKEN").replace('"', '')
    base_url = config.get("Parameters", "DATA_SERVER")
    client   = Client(api_key=api_key, base_url=base_url)

    # --- Load checkpoint (empty if rebuild or first run) --------------------
    existing_df, done_ids = _load_checkpoint()

    # --- Step 1 : collect all unique station IDs ----------------------------
    logger.info("=" * 60)
    logger.info("Step 1 — Listing stations across %d departments", len(DEPARTMENTS))
    logger.info("=" * 60)

    station_ids: set[str] = set()
    for i, dept in enumerate(DEPARTMENTS, 1):
        logger.info("  [%2d/%d] department %-4s", i, len(DEPARTMENTS), dept)
        stations = _fetch_dept_stations(client, dept)
        time.sleep(REQUEST_INTERVAL)
        for s in stations:
            station_ids.add(str(s['id']))

    todo_ids = sorted(station_ids - done_ids)
    logger.info(
        "Found %d unique stations total — %d already fetched — %d to fetch.",
        len(station_ids), len(done_ids), len(todo_ids),
    )

    if not todo_ids:
        logger.info("Nothing to do — %s is already complete.", OUTPUT_CSV)
        return

    # --- Step 2 : fetch station details -------------------------------------
    eta_min = len(todo_ids) * REQUEST_INTERVAL / 60
    logger.info("=" * 60)
    logger.info(
        "Step 2 — Fetching details for %d stations  (ETA ≈ %.0f min)",
        len(todo_ids), eta_min,
    )
    logger.info("=" * 60)

    pending_rows: list[dict] = []

    for i, sid in enumerate(todo_ids, 1):
        row = _fetch_station_info(client, sid)
        time.sleep(REQUEST_INTERVAL)

        if row:
            pending_rows.append(row)
        else:
            logger.warning("  [%d/%d] station %s — skipped (no data).", i, len(todo_ids), sid)

        # Progress log
        if i % 50 == 0 or i == len(todo_ids):
            elapsed  = (datetime.now() - t0).seconds
            pct      = 100 * i / len(todo_ids)
            rem_min  = (len(todo_ids) - i) * REQUEST_INTERVAL / 60
            logger.info(
                "  [%d/%d]  %.0f%%  —  elapsed %dm%02ds  —  remaining ≈ %.0f min",
                i, len(todo_ids), pct, elapsed // 60, elapsed % 60, rem_min,
            )

        # Checkpoint save
        if pending_rows and i % CHECKPOINT_EVERY == 0:
            existing_df = _save(pending_rows, existing_df)
            logger.info("  ✓ Checkpoint saved (%d stations written so far).", len(existing_df))
            pending_rows = []

    # Final save
    if pending_rows:
        existing_df = _save(pending_rows, existing_df)

    elapsed = (datetime.now() - t0).seconds
    logger.info("=" * 60)
    logger.info(
        "Done — %d stations saved to %s  (total time %dm%02ds).",
        len(existing_df), OUTPUT_CSV, elapsed // 60, elapsed % 60,
    )
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
