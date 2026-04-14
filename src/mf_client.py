"""
HTTP client for the Météo-France DPClim public API (hourly data).

Supports token-based authentication (apikey header) and OAuth2
client-credentials flow.  On a 401 response the client automatically
retries with a fresh OAuth2 token when an application_id is configured.

Author:  Samy KRAIEM
Created: 2024
Updated: 2026
"""

import logging
import math
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TOKEN_URL        = "https://portail-api.meteofrance.fr/token"
REQUEST_INTERVAL = 0.7  # seconds between requests to stay under 100 req/min


def _parse_retry_wait(response: requests.Response, default: int = 65) -> int:
    """Return how many seconds to wait after a 429 response.

    Tries to read ``nextAccessTime`` from the JSON body and compute the
    exact delay.  Falls back to *default* seconds if parsing fails.

    Args:
        response: The 429 HTTP response from the API.
        default:  Fallback wait time in seconds.

    Returns:
        int: Number of seconds to sleep before the next attempt.
    """
    try:
        next_access = response.json().get('nextAccessTime', '')
        # Format observed: "2026-avr.-14 23:21:06+0000 UTC"
        # Strip the trailing " UTC" and parse the ISO-like prefix.
        clean = next_access.replace(' UTC', '').strip()
        # Replace French abbreviated months with numbers so strptime works.
        _FR_MONTHS = {
            'janv.': '01', 'févr.': '02', 'mars':  '03', 'avr.':  '04',
            'mai':   '05', 'juin':  '06', 'juil.': '07', 'août':  '08',
            'sept.': '09', 'oct.':  '10', 'nov.':  '11', 'déc.':  '12',
        }
        for fr, num in _FR_MONTHS.items():
            clean = clean.replace(fr, num)
        # Expected after replacement: "2026-04-14 23:21:06+0000"
        dt = datetime.strptime(clean, '%Y-%m-%d %H:%M:%S%z')
        wait = math.ceil((dt - datetime.now(timezone.utc)).total_seconds()) + 2
        if wait > 0:
            logger.info("Rate-limit window ends at %s — waiting %ds.", dt.isoformat(), wait)
            return wait
    except Exception:
        pass
    return default

class Client(object):
    """Météo-France DPClim API client supporting token and OAuth2 auth."""

    def __init__(self, api_key=None, application_id=None, base_url=None):
        """Initialise the client and configure authentication headers.

        Args:
            api_key (str, optional): Static API token used in the
                ``apikey`` request header.
            application_id (str, optional): Base-64 client credentials
                string for OAuth2 client-credentials flow.
            base_url (str, optional): DPClim API base URL.

        Raises:
            ValueError: If neither *api_key* nor *application_id* is
                provided.
        """
        if not api_key and not application_id:
            raise ValueError("Either an Token or an Application ID must be provided.")
        self.base_url = base_url if base_url is not None else "DEFAULT_BASE_URL"
        self.session = requests.Session()
        self.api_key = api_key
        self.application_id = application_id
        if api_key:
            self.session.headers.update({'apikey': self.api_key})
        else:
            self.obtain_oauth2_token()

    def request(self, method, url, **kwargs):
        response = self.session.request(method, url, **kwargs)
        if response.status_code == 401:
            if self.application_id:
                self.obtain_oauth2_token()
                response = self.session.request(method, url, **kwargs)
            else:
                logger.error("Token is out of date and no Application ID was supplied. Update API_config.txt with valid credentials.")
        return response

    def obtain_oauth2_token(self):
        """Fetch a fresh OAuth2 bearer token and update the session header.

        Raises:
            RuntimeError: If the token endpoint request fails or the
                response does not contain an access_token.
        """
        data = {'grant_type': 'client_credentials'}
        headers = {'Authorization': 'Basic ' + self.application_id}
        try:
            access_token_response = requests.post(TOKEN_URL, data=data, headers=headers)
            access_token_response.raise_for_status()
            token = access_token_response.json()['access_token']
        except (requests.RequestException, KeyError, ValueError) as e:
            raise RuntimeError(f"OAuth2 authentication failed. Check APPLICATION_ID in API_config.txt. ({e})")
        self.session.headers.update({'Authorization': 'Bearer %s' % token})

    def get_stations_list(self, DEPARTEMENT):
        """Return the list of hourly stations for a given department.

        Args:
            DEPARTEMENT (str): French department number (e.g. ``"75"``).

        Returns:
            list: JSON list of station dicts, or None on HTTP error.
        """
        stations_url = self.base_url + "/liste-stations/horaire?id-departement=" + str(DEPARTEMENT)
        response = self.request('GET', stations_url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error("get_stations_list() : Unexpected status code %s: %s", response.status_code, response.text)
            return None

    def get_station_info(self, station_id):
        """Retrieve metadata for a single station and return it as a DataFrame.

        Args:
            station_id (str): DPClim station identifier.

        Returns:
            pd.DataFrame: One-row DataFrame with columns ID, Nom, LieuDit,
                Bassin, DateDebut, DateFin, Type, Altitude, Latitude,
                Longitude.  Returns None on HTTP error.
        """
        station_info_url = self.base_url + "/information-station?id-station=" + str(station_id)
        response = self.request('GET', station_info_url)
        if response.status_code == 200:
            data = response.json()
            # JSON to DataFrame
            ids, noms, lieuxDits, bassins, datesDebut, datesFin, types, altitudes, latitudes, longitudes = [], [], [], [], [], [], [], [], [], []
            for item in data:
                # General informations
                ids.append(item.get('id'))
                noms.append(item.get('nom'))
                lieuxDits.append(item.get('lieuDit'))
                bassins.append(item.get('bassin'))
                datesDebut.append(item.get('dateDebut'))
                datesFin.append(item.get('dateFin'))
                if 'typesPoste' in item and item['typesPoste']:
                    types.append(item['typesPoste'][0].get('type'))
                else:
                    types.append(None)
                if 'positions' in item and item['positions']:
                    position = item['positions'][0]
                    altitudes.append(position.get('altitude'))
                    latitudes.append(position.get('latitude'))
                    longitudes.append(position.get('longitude'))
                else:
                    altitudes.append(None)
                    latitudes.append(None)
                    longitudes.append(None)
            df = pd.DataFrame({
                'ID': ids,
                'Nom': noms,
                'LieuDit': lieuxDits,
                'Bassin': bassins,
                'DateDebut': datesDebut,
                'DateFin': datesFin,
                'Type': types,
                'Altitude': altitudes,
                'Latitude': latitudes,
                'Longitude': longitudes
            })
            return df
        else:
            return None

    def order_station_data(self, station_id, start_year, end_year):
        """Place one data order per year and return the list of order IDs.

        The DPClim API limits each order to a single calendar year, so
        this method loops from *start_year* to *end_year* inclusive.

        Args:
            station_id (str): DPClim station identifier.
            start_year (int): First year to order (inclusive).
            end_year (int): Last year to order (inclusive).

        Returns:
            list: Order ID strings for successfully placed orders.
        """
        order_ids  = []
        max_retries = 5
        order_url_tmpl = (
            self.base_url
            + "/commande-station/horaire"
            + "?id-station={sid}"
            + "&date-deb-periode={y}-01-01T00%3A00%3A00Z"
            + "&date-fin-periode={y}-12-31T23%3A00%3A00Z"
        )

        for year in range(int(start_year), int(end_year) + 1):
            url = order_url_tmpl.format(sid=station_id, y=year)

            for attempt in range(1, max_retries + 1):
                if attempt == 1:
                    logger.info("Placing order for the year %d…", year)
                else:
                    logger.info("Placing order for the year %d — retry %d/%d…",
                                year, attempt, max_retries)
                response = self.request('GET', url)
                time.sleep(REQUEST_INTERVAL)

                if response.status_code == 202:
                    try:
                        order_id = response.json()[
                            'elaboreProduitAvecDemandeResponse']['return']
                        order_ids.append(order_id)
                        logger.info("Year %d — order placed: %s", year, order_id)
                    except (ValueError, KeyError):
                        logger.error("Year %d — could not extract order ID from response.",
                                     year)
                    break  # success or unrecoverable parse error — move to next year

                elif response.status_code == 429:
                    wait = _parse_retry_wait(response, default=65)
                    if attempt < max_retries:
                        logger.warning(
                            "Year %d — rate limited (429), attempt %d/%d, "
                            "waiting %ds before retry.",
                            year, attempt, max_retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "Year %d — rate limit persists after %d attempts, skipping.",
                            year, max_retries,
                        )

                else:
                    logger.error("Year %d — unexpected status %s: %s",
                                 year, response.status_code, response.text[:200])
                    break  # non-retryable error

        return order_ids

    def download_command_file(self, order_ids):
        """Poll and download the CSV file for each order ID.

        Polls with up to 10 retry attempts per order:
        - 201 — file ready, download and save as
          ``command_{order_id}_RAW_DATA.csv``.
        - 204 — not ready yet, wait 10 s.
        - 429 — rate-limited, wait 60 s.
        - 500 — server-side processing, wait 60 s.
        - 404 / 410 / 507 — permanent failure, skip.

        Args:
            order_ids (list): Order ID strings returned by
                order_station_data().
        """
        for order_id in order_ids:
            url = f"{self.base_url}/commande/fichier?id-cmde={order_id}"
            ready = False
            attempt = 0
            max_attempts = 10

            while not ready and attempt < max_attempts:
                response = self.request('GET', url)
                time.sleep(REQUEST_INTERVAL)
                if response.status_code == 201:
                    filename = f"command_{order_id}_RAW_DATA.csv"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    logger.info("Order %s downloaded → %s", order_id, filename)
                    ready = True
                elif response.status_code == 429:
                    attempt += 1
                    logger.warning("Order %s — rate limited (429), attempt %d/%d, waiting 60s.", order_id, attempt, max_attempts)
                    time.sleep(60 - REQUEST_INTERVAL)  # REQUEST_INTERVAL already elapsed above
                elif response.status_code in [204, 500]:
                    attempt += 1
                    wait_time = 10 if response.status_code == 204 else 60
                    logger.warning("Order %s — status %s, attempt %d/%d, waiting %ds.", order_id, response.status_code, attempt, max_attempts, wait_time)
                    time.sleep(wait_time - REQUEST_INTERVAL)  # REQUEST_INTERVAL already elapsed above
                elif response.status_code in [404, 410, 507]:
                    logger.error("Order %s — download failed with status %s: %s", order_id, response.status_code, response.text)
                    break
                else:
                    logger.error("Order %s — unexpected status %s: %s", order_id, response.status_code, response.text)
                    break
            if attempt == max_attempts:
                logger.error("Order %s — maximum retry attempts (%d) reached.", order_id, max_attempts)
