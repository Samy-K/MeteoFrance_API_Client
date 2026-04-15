"""
Shared utility functions used across multiple modules.

Provides statistical (Mann-Kendall), astronomical (solar elevation),
coordinate-parsing, and matplotlib styling helpers.

Author:  Samy KRAIEM
Created: 2024
Updated: 2026
"""

import math
import os
import re
import sys
from typing import Tuple

import numpy as np


def resource_path(relative_path: str) -> str:
    """Return the absolute path to a bundled resource.

    In development, resolves relative to the current working directory.
    When frozen by PyInstaller (--onefile), resolves relative to the
    temporary extraction directory ``sys._MEIPASS``.

    Args:
        relative_path (str): Path relative to the project root
            (e.g. ``'weather_stations_infos/mf_stations.csv'``).

    Returns:
        str: Absolute path usable with ``open()`` or ``pd.read_csv()``.
    """
    base = getattr(sys, '_MEIPASS', os.getcwd())
    return os.path.join(base, relative_path)


def mann_kendall(y) -> Tuple[float, float, float, float]:
    """Compute the Mann-Kendall trend test and Theil-Sen estimator.

    Operates on an annual time series.  No external dependencies —
    uses math.erf for the normal CDF.  Ties correction is omitted
    as it is negligible on annual climate series.

    Args:
        y (array-like): 1-D sequence of annual values (NaNs are dropped).

    Returns:
        tuple: (tau, p_value, sen_slope_per_year, sen_intercept).
            All four values are NaN when fewer than 4 valid data points
            are available.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = y[j] - y[i]
            s += 1 if d > 0 else (-1 if d < 0 else 0)

    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p   = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    tau = s / (n * (n - 1) / 2.0)

    slopes = [
        (y[j] - y[i]) / (j - i)
        for i in range(n - 1) for j in range(i + 1, n)
    ]
    sen       = float(np.median(slopes))
    intercept = float(np.median(y) - sen * np.median(np.arange(n)))
    return tau, p, sen, intercept


def solar_elevation_vec(lat_deg: float, lon_deg: float,
                        dti: 'pd.DatetimeIndex') -> 'np.ndarray':
    """Return the vectorised solar elevation angle for each UTC timestamp.

    Uses simplified solar geometry (±0.5° accuracy), which is sufficient
    for day/night detection.

    Args:
        lat_deg (float): Station latitude in decimal degrees.
        lon_deg (float): Station longitude in decimal degrees.
        dti (pd.DatetimeIndex): UTC timestamps to evaluate.

    Returns:
        np.ndarray: Solar elevation angle in degrees for each timestamp.
    """
    doy      = dti.dayofyear.values.astype(float)
    hour_utc = dti.hour.values + dti.minute.values / 60.0

    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + doy) / 365.0)))
    B    = np.radians(360.0 * (doy - 81.0) / 364.0)
    eot  = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)  # minutes

    solar_time = hour_utc + lon_deg / 15.0 + eot / 60.0
    hour_angle = np.radians(15.0 * (solar_time - 12.0))

    lat      = math.radians(lat_deg)
    sin_elev = (math.sin(lat) * np.sin(decl) +
                math.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


def parse_coord(s: str) -> float:
    """Parse a coordinate string in decimal or cardinal-degree notation.

    Accepted formats::

        48.85   48,85   -2.35   +48.85
        44,38°N   44.38°N   44,38° N
         4,64°E    4.64°E    4,64° E
        44,38°S  →  -44.38
         4,64°W  →   -4.64

    Args:
        s (str): Coordinate string to parse.

    Returns:
        float: Coordinate value in decimal degrees (negative for S/W).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    s = s.strip()
    m = re.match(r'^([+-]?\d+[.,]?\d*)\s*°\s*([NSEWnsew])$', s)
    if m:
        val  = float(m.group(1).replace(',', '.'))
        card = m.group(2).upper()
        return -val if card in ('S', 'W') else val
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        raise ValueError(f"Cannot parse coordinate: '{s}'")


def label_slug(label: str) -> str:
    """Convert a variable label to a filesystem-safe slug.

    Example::

        label_slug('2 m Air Temperature') → '2_m_air_temperature'
        label_slug('Dew Point Temperature') → 'dew_point_temperature'

    Args:
        label (str): Human-readable variable label.

    Returns:
        str: Lowercase slug with non-alphanumeric runs replaced by ``_``.
    """
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def save_ax(fig, ax, path: str, dpi: int = 300) -> None:
    """Save a single axes panel from a multi-axes figure as a PNG.

    Extracts the tight bounding box of *ax* (including its title and labels)
    from the already-drawn *fig* and writes it to *path* at *dpi* dpi.
    The output directory is created automatically if it does not exist.

    Args:
        fig: The parent matplotlib Figure.
        ax: The Axes to extract.
        path (str): Destination file path (PNG).
        dpi (int): Resolution in dots per inch (default 300).
    """
    import os
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
        bb_disp = ax.get_tightbbox(renderer)
    except AttributeError:
        bb_disp = ax.get_tightbbox()
    if bb_disp is None:
        return
    bb_in = bb_disp.transformed(fig.dpi_scale_trans.inverted())
    bb_in = bb_in.expanded(1.03, 1.06)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, bbox_inches=bb_in, dpi=dpi, facecolor='white')


def style_table(tbl, header_color: str = '#2c3e50', stripe_color: str = '#ecf0f1',
                fontsize: float = 8, scale: float = 1.4) -> None:
    """Apply consistent dark-header and alternating-row styling to a matplotlib table.

    Args:
        tbl: Matplotlib Table object to style.
        header_color (str): Background colour for the header row.
        stripe_color (str): Background colour for even data rows.
        fontsize (float): Font size for all cells.
        scale (float): Row-height scale factor passed to Table.scale().
    """
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, scale)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor(stripe_color)
