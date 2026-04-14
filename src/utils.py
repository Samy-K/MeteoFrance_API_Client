"""
Shared helpers used by multiple modules.
"""

import math
import re
from typing import Tuple

import numpy as np


def mann_kendall(y) -> Tuple[float, float, float, float]:
    """
    Mann-Kendall trend test + Theil-Sen estimator on an annual series.
    Returns (tau, p_value, sen_slope_per_year, sen_intercept).

    No external dependencies (uses math.erf for the normal CDF).
    Ties correction omitted — negligible on annual climate series.
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
    """
    Vectorized solar elevation angle (degrees) for each UTC timestamp in dti.
    Uses simplified solar geometry (±0.5° accuracy — sufficient for day/night detection).
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
    """
    Parse a coordinate string in decimal or cardinal-degree notation.

    Accepted examples:
      48.85      48,85      -2.35      +48.85
      44,38°N    44.38°N    44,38° N
       4,64°E     4.64°E     4,64° E
      44,38°S   →  -44.38
       4,64°W   →   -4.64

    Raises ValueError if the string cannot be parsed.
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


def style_table(tbl, header_color: str = '#2c3e50', stripe_color: str = '#ecf0f1',
                fontsize: float = 8, scale: float = 1.4) -> None:
    """Apply consistent dark-header / alternating-row styling to a matplotlib table."""
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
