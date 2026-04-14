# MeteoFrance API Client

A Python tool for downloading, processing, and reporting hourly surface
meteorological data from two complementary sources:

- **Météo-France DPClim** — French national network (~2 000 stations,
  French metropolitan territory + overseas).
- **NOAA ISD-Lite** — NCEI Integrated Surface Database (~30 000 stations
  worldwide, hourly data back to the 1970s for major airports).

For each selected station the tool produces:

1. A commented CSV file with the processed hourly dataset.
2. A multi-page quality-control PDF (`quality_checks.pdf`).
3. A climatological factsheet PDF (`factsheet.pdf`).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [First-time setup — station catalogues](#first-time-setup--station-catalogues)
- [Usage](#usage)
- [Output files](#output-files)
- [Architecture](#architecture)
- [Data sources and variables](#data-sources-and-variables)
- [Quality control methodology](#quality-control-methodology)
- [License](#license)

---

## Requirements

- Python ≥ 3.10
- [`pandas`](https://pandas.pydata.org/) ≥ 2.0
- [`requests`](https://docs.python-requests.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`numpy`](https://numpy.org/)

No optional dependencies — Mann-Kendall and solar geometry are implemented
from scratch.

---

## Installation

```bash
git clone https://github.com/Samy-K/MF_API_CLIENT.git
cd MF_API_CLIENT
pip install -r requirements.txt
```

---

## Configuration

Create (or update) `API_config.txt` in the project root **before** using
the Météo-France data source:

```ini
[Parameters]
APPLICATION_ID = Your_Application_ID   ; OAuth2 client-credentials (valid 1 h)
DATA_SERVER    = https://public-api.meteofrance.fr/public/DPClim/v1
TOKEN          = Your_API_Key          ; Static API token
```

> Provide **at least one** of `APPLICATION_ID` or `TOKEN`.  
> When a static token expires (HTTP 401), the client automatically fetches a
> fresh OAuth2 token if `APPLICATION_ID` is set.

API credentials and documentation:
<https://portail-api.meteofrance.fr/web/fr>

The NOAA ISD-Lite source requires **no credentials** — data is fetched
directly from the NCEI public HTTPS server.

---

## First-time setup — station catalogues

Two local catalogue files are required before the main application can run.

### Météo-France stations (~60 min)

Queries the DPClim API for all ~4 500 stations across all departments:

```bash
python src/fetch_mf_stations.py
```

Output: `weather_stations_infos/mf_stations.csv`

The script supports **resume/checkpoint**: if interrupted, re-running it
skips already-fetched stations and appends new rows.

### NOAA ISD stations (a few seconds)

Downloads the public NOAA ISD catalogue in a single HTTP request:

```bash
python src/fetch_isd_stations.py
```

Output: `weather_stations_infos/isd_stations.csv`

Re-run at any time to refresh the catalogue.

---

## Usage

```bash
python main.py
```

The script is fully interactive:

```
  Select data source:
    [1]  Météo-France  (DPClim API — French stations)
    [2]  NOAA ISD-Lite (Integrated Surface Database — worldwide)
```

### Météo-France flow

1. Choose a station selection mode:
   - **By department** — lists all stations in a French department.
   - **By station ID** — direct entry of a known DPClim ID.
   - **By name** — substring search (requires `mf_stations.csv`).
   - **By coordinates** — finds the 10 nearest stations (requires `mf_stations.csv`).
2. Confirm the year range.
3. Confirm the order placement.
4. The script places one order per year, polls for readiness, and downloads
   the CSV files.

### NOAA ISD flow

1. Choose a station search mode:
   - **By name** — case-insensitive substring search.
   - **By coordinates** — finds the 10 nearest stations (Haversine distance).
2. Confirm the year range.
3. The script downloads and decompresses one `.gz` file per year directly
   into memory — no intermediate files.

### Coordinate input formats

The coordinate prompts accept any of the following formats:

```
48.85       48,85       -2.35       +48.85
44,38°N     44.38°N     44,38° N
4,64°E       4.64°E      4,64° E
44,38°S  →  -44.38      4,64°W  →   -4.64
```

---

## Output files

All outputs are written to a directory named `out_{station_id}_{station_name}/`:

| File | Description |
|------|-------------|
| `RAW_DATA_{id}_{name}_{start}-{end}.csv` | Processed hourly dataset with `#`-prefixed station metadata header and per-variable quality-flag columns (`FLAG_T`, `FLAG_FF`, …). |
| `quality_checks.pdf` | Multi-page quality-control report (cover + coherence page + one page per variable). |
| `factsheet.pdf` | Climatological factsheet (cover + one page per variable group). |
| `figures/*.png` | Individual figure exports (one per PDF page, 150 dpi). |

---

## Architecture

```
main.py                     Entry point — orchestrates the full workflow
src/
  mf_client.py              Météo-France DPClim API client (token + OAuth2)
  isd_client.py             NOAA ISD-Lite HTTPS client
  data_handler.py           DatasetManager — read, subset, flag, save
  data_quality_report.py    Quality-control PDF generator
  data_facts_report.py      Climatological factsheet PDF generator
  station_selector.py       Interactive station search and selection (CLI I/O)
  utils.py                  Shared helpers (Mann-Kendall, solar geometry, …)
  fetch_mf_stations.py      One-time script — builds mf_stations.csv
  fetch_isd_stations.py     One-time script — builds isd_stations.csv
weather_stations_infos/
  mf_stations.csv           Météo-France station catalogue (generated)
  isd_stations.csv          NOAA ISD station catalogue (generated)
API_config.txt              Météo-France API credentials (not committed)
```

### Data flow

```
API_config.txt / isd_stations.csv / mf_stations.csv
        ↓
  mf_client.Client  OR  isd_client.ISDClient
        ↓
  station selection  (department / ID / name / coordinates)
        ↓
  [MF]  order_station_data → download_command_file → command_{id}_RAW_DATA.csv
  [ISD] download_and_parse → DatasetManager (in memory)
        ↓
  DatasetManager.create_subset → add_quality_flags → save_subset_as_csv
        ↓
  out_{id}_{name}/RAW_DATA_*.csv
        ↓
  generate_quality_pdf  →  out_*/quality_checks.pdf
  generate_factsheet_pdf →  out_*/factsheet.pdf
```

---

## Data sources and variables

### Météo-France DPClim — variables available

| Code | Description | Unit |
|------|-------------|------|
| `T` | 2 m air temperature | °C |
| `TD` | Dew-point temperature | °C |
| `U` | Relative humidity | % |
| `UABS` | Absolute humidity | g/m³ |
| `PSTAT` | Station pressure | hPa |
| `GLO` | Global solar radiation | W/m² |
| `DIR` | Direct solar radiation | W/m² |
| `DIF` | Diffuse solar radiation | W/m² |
| `INFRAR` | Downwelling infrared radiation | W/m² |
| `N` | Cloud cover | oktas |
| `DD` | Wind direction | ° |
| `FF` | Wind speed | m/s |
| `RR1` | Hourly precipitation | mm |

Full field documentation: <https://portail-api.meteofrance.fr/web/en/>


### NOAA ISD-Lite — variables available

| Code | Description | Unit |
|------|-------------|------|
| `T` | 2 m air temperature | °C |
| `TD` | Dew-point temperature | °C |
| `U` | Relative humidity (derived) | % |
| `UABS` | Absolute humidity (derived) | g/m³ |
| `PSTAT` | Sea-level pressure (SLP) | hPa |
| `DD` | Wind direction | ° |
| `FF` | Wind speed | m/s |
| `N` | Sky cover | oktas |
| `RR1` | Hourly liquid precipitation | mm |

`U` and `UABS` are derived from `T` and `TD` using the August-Roche-Magnus
approximation. Radiation variables (`GLO`, `DIR`, `DIF`, `INFRAR`) are not
available in ISD-Lite — the factsheet radiation page is skipped automatically.

---

## Quality control methodology

Five flag types are applied to each variable independently:

| Flag | Description |
|------|-------------|
| `MISSING` | Timestamp present but no measurement recorded. |
| `ABSURD` | Value outside variable-specific physical bounds. |
| `JUMP` | Hourly change exceeds the maximum allowed delta. |
| `PLATEAU` | Too many consecutive identical values (sensor freeze). |
| `CONTEXTUAL` | Statistical outlier vs. typical (month, hour) distribution via Q1 − 3 × IQR / Q3 + 3 × IQR. |

Cross-variable coherence checks are also performed:

| Check | Condition |
|-------|-----------|
| TD > T | Dew point above air temperature — thermodynamically impossible. |
| GLO > DIR + DIF | Global radiation exceeds component sum by more than 20 W/m². |
| RR1 > 0, U = 0 | Precipitation recorded with zero relative humidity. |
| GLO at night | Global radiation > 5 W/m² during astronomical night. |

Flag codes are stored in `FLAG_{var}` columns in the output CSV.  Multiple
flags on the same observation are separated by `|` (e.g. `ABSURD|CONTEXTUAL`).

---

## License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for details.

---

*Author: Samy KRAIEM — 2024–2026*
