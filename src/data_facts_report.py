"""
Climatological factsheet generator - MeteoFrance hourly data.
Produces factsheet.pdf (one page per variable group) + PNG figures.

Pages:
  - Cover        station info + full-period climatological normals
  - Temperature  T (+ TD overlay)  — monthly clim, diurnal cycle, annual trend, records
  - Humidity     U, UABS           — monthly clim, diurnal cycle, annual trend, records
  - Pressure     PSTAT             — monthly clim, diurnal cycle, annual trend, records
  - Wind         FF + DD           — wind rose, monthly boxplot, annual trend, records
  - Radiation    GLO/DIR/DIF/INFRAR— monthly clim, diurnal heatmap, annual trend
  - Precipitation RR1              — monthly cumul clim, annual total trend, extreme events
  - Cloud Cover  N                 — monthly okta frequency, annual trend

Mann-Kendall test and Theil-Sen slope are implemented without external dependencies.
A warning is emitted if the record is shorter than MIN_YEARS_WARNING years.
"""

import logging
import math
import os
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

MIN_YEARS_WARNING = 20

_MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_SEASONS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}
_SEASON_COLORS = {
    'DJF': '#5b9bd5',
    'MAM': '#70ad47',
    'JJA': '#ed7d31',
    'SON': '#8e44ad',
}


# ── Mann-Kendall (no external dependencies) ────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mann_kendall(y) -> Tuple[float, float, float, float]:
    """
    Mann-Kendall trend test + Theil-Sen estimator on an annual series.
    Returns (tau, p_value, sen_slope_per_year, sen_intercept).
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

    p   = 2.0 * (1.0 - _norm_cdf(abs(z)))
    tau = s / (n * (n - 1) / 2.0)

    slopes = [
        (y[j] - y[i]) / (j - i)
        for i in range(n - 1) for j in range(i + 1, n)
    ]
    sen       = float(np.median(slopes))
    intercept = float(np.median(y) - sen * np.median(np.arange(n)))
    return tau, p, sen, intercept


def _mk_text(tau: float, p: float, slope: float, unit: str) -> str:
    if any(np.isnan(v) for v in [tau, p, slope]):
        return "Mann-Kendall: insufficient data (< 4 years)"
    arrow = '↑' if slope > 0 else ('↓' if slope < 0 else '→')
    sig   = "p < 0.05  ✓" if p < 0.05 else f"p = {p:.2f}  (not significant)"
    return (
        f"Mann-Kendall   τ = {tau:.3f}    {sig}\n"
        f"Theil-Sen slope:  {arrow} {abs(slope * 10):.4f} {unit}/decade"
    )


# ── Style helpers ──────────────────────────────────────────────────────────────

def _style_table(tbl, header_color: str = '#2c3e50', stripe: str = '#ecf0f1'):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor(stripe)


def _mk_box(ax, text: str, p: float):
    """Annotate axes with a coloured Mann-Kendall result box."""
    color = '#c0392b' if (not np.isnan(p) and p < 0.05) else '#7f8c8d'
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=7, va='top',
            linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.4', fc=color, alpha=0.13, ec=color))


# ── Data helpers ───────────────────────────────────────────────────────────────

def _annual_agg(series: pd.Series, dti: pd.DatetimeIndex,
                agg: str = 'mean') -> pd.Series:
    s = pd.Series(series.values, index=dti)
    r = s.resample('YE')
    if agg == 'mean': return r.mean().dropna()
    if agg == 'sum':  return r.sum(min_count=1).dropna()
    if agg == 'max':  return r.max().dropna()
    if agg == 'min':  return r.min().dropna()
    return r.mean().dropna()


def _top5(series: pd.Series, dates: pd.Series, mode: str = 'max') -> pd.DataFrame:
    """Return Top-5 max or min records with their timestamp."""
    valid = series.dropna()
    if valid.empty:
        return pd.DataFrame(columns=['Rank', 'Value', 'Date'])
    idx = valid.nlargest(5).index if mode == 'max' else valid.nsmallest(5).index
    return pd.DataFrame({
        'Rank':  range(1, len(idx) + 1),
        'Value': valid.loc[idx].values,
        'Date':  dates.loc[idx].dt.strftime('%Y-%m-%d %Hh').values,
    })


def _grp_month(series: pd.Series, dti: pd.DatetimeIndex) -> pd.core.groupby.SeriesGroupBy:
    """Group a series by calendar month (1–12) using dti.month."""
    tmp = pd.DataFrame({'v': series.values, 'm': np.array(dti.month)})
    return tmp.groupby('m')['v']


def _trend_ax(ax, ann: pd.Series, tau, p, slope, intercept, unit: str):
    years = ann.index.year.values
    vals  = ann.values
    ax.scatter(years, vals, color='#2c3e50', s=20, zorder=3, alpha=0.85)
    ax.plot(years, vals, color='#2c3e50', lw=0.8, alpha=0.40)
    if not np.isnan(slope):
        n     = len(years)
        trend = slope * np.arange(n) + intercept
        c     = '#c0392b' if (not np.isnan(p) and p < 0.05) else '#95a5a6'
        ax.plot(years, trend, color=c, lw=2.0, ls='--', zorder=4)
    step = max(1, len(years) // 10)
    ax.set_xticks(years[::step])
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(axis='y', alpha=0.25)


def _diurnal_ax(ax, series: pd.Series, dti: pd.DatetimeIndex, unit: str):
    """Mean diurnal cycle ± 1σ for each meteorological season."""
    months = np.array(dti.month)
    hours  = np.array(dti.hour)
    vals   = series.values.astype(float)
    for seas, mths in _SEASONS.items():
        mask = np.isin(months, mths)
        tmp  = pd.DataFrame({'v': vals[mask], 'h': hours[mask]})
        g    = tmp.groupby('h')['v']
        mu, sd = g.mean(), g.std()
        ax.plot(mu.index, mu.values, color=_SEASON_COLORS[seas],
                lw=1.8, label=seas)
        ax.fill_between(mu.index, mu - sd, mu + sd,
                        color=_SEASON_COLORS[seas], alpha=0.10)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)], fontsize=6)
    ax.set_xlabel('Hour (UTC)', fontsize=7)
    ax.set_ylabel(unit, fontsize=8)
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.7)
    ax.grid(alpha=0.20)
    ax.tick_params(axis='y', labelsize=7)


def _records_panel(ax, rec_max: pd.DataFrame, rec_min: pd.DataFrame,
                   col_max: str, col_min: str, unit: str):
    """Two side-by-side record tables (max left, min right)."""
    ax.axis('off')
    fmt = lambda v: f"{v:.2f} {unit}"
    rows_max = [[str(r.Rank), fmt(r.Value), r.Date] for r in rec_max.itertuples()] \
               if not rec_max.empty else []
    rows_min = [[str(r.Rank), fmt(r.Value), r.Date] for r in rec_min.itertuples()] \
               if not rec_min.empty else []
    if rows_max:
        t = ax.table(cellText=rows_max, colLabels=['#', col_max, 'Date'],
                     bbox=[0, 0, 0.46, 1], cellLoc='center')
        _style_table(t, header_color='#c0392b')
    if rows_min:
        t = ax.table(cellText=rows_min, colLabels=['#', col_min, 'Date'],
                     bbox=[0.54, 0, 0.46, 1], cellLoc='center')
        _style_table(t, header_color='#1a5276')


# ── Wind rose ──────────────────────────────────────────────────────────────────

_SPD_BINS   = [0, 2, 5, 10, 15, np.inf]
_SPD_LABELS = ['< 2', '2-5', '5-10', '10-15', '> 15']
_SPD_COLORS = ['#d4e6f1', '#85c1e9', '#2980b9', '#1a5276', '#0b2545']
_N_SECTORS  = 16


def _wind_rose_ax(ax, ff: pd.Series, dd: pd.Series):
    """Draw a 16-sector wind rose on a polar axes (must use projection='polar')."""
    n_tot = int(ff.notna().sum())
    if n_tot == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    valid = ff.notna() & dd.notna()
    ff_v  = ff[valid].values
    dd_v  = dd[valid].values

    sec_deg = 360.0 / _N_SECTORS
    centers = np.arange(0, 360, sec_deg)      # sector centres, degrees CW from N
    thetas  = np.deg2rad(centers)             # for polar (N-top, CW after set_theta_*)
    width   = 2 * np.pi / _N_SECTORS * 0.88
    bottom  = np.zeros(_N_SECTORS)

    for lo, hi, label, color in zip(_SPD_BINS[:-1], _SPD_BINS[1:],
                                     _SPD_LABELS, _SPD_COLORS):
        spd_mask = (ff_v >= lo) & (ff_v < hi)
        heights  = np.zeros(_N_SECTORS)
        for si, cen in enumerate(centers):
            lo_d = (cen - sec_deg / 2) % 360
            hi_d = (cen + sec_deg / 2) % 360
            if lo_d < hi_d:
                dir_mask = (dd_v >= lo_d) & (dd_v < hi_d)
            else:                                    # sector straddles 360 / 0
                dir_mask = (dd_v >= lo_d) | (dd_v < hi_d)
            heights[si] = 100.0 * (spd_mask & dir_mask).sum() / n_tot
        ax.bar(thetas, heights, width=width, bottom=bottom,
               color=color, edgecolor='white', linewidth=0.3, label=f'{label} m/s')
        bottom += heights

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(alpha=0.3)

    n_calm   = int((ff.dropna() < 1).sum())
    pct_calm = 100.0 * n_calm / n_tot
    ax.set_title(f'Calm (FF < 1 m/s): {pct_calm:.1f}%', fontsize=7.5, pad=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Cover page
# ═══════════════════════════════════════════════════════════════════════════════

def _cover_page(pdf: PdfPages, df: pd.DataFrame, station_info: pd.DataFrame,
                n_years: float, dti: pd.DatetimeIndex,
                figures_dir: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.975, "Climatological Factsheet",
             ha='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.951,
             f"{station_info.iloc[0].get('ID', '')}  –  "
             f"{station_info.iloc[0].get('Nom', '')}",
             ha='center', fontsize=13, color='#333')
    fig.text(0.5, 0.930,
             f"{df['DATE'].min().strftime('%d/%m/%Y')}  →  "
             f"{df['DATE'].max().strftime('%d/%m/%Y')}",
             ha='center', fontsize=11, color='#555')
    fig.text(0.5, 0.912,
             f"Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}",
             ha='center', fontsize=9, color='#777')

    y_cur = 0.893
    if n_years < MIN_YEARS_WARNING:
        fig.text(0.5, y_cur,
                 f"Warning — only {n_years:.0f} years of data: "
                 "climatological statistics may not be representative.",
                 ha='center', fontsize=8.5, color='#c0392b', fontweight='bold')
        y_cur -= 0.020

    # Station info
    fig.text(0.06, y_cur, 'Station Information',
             fontsize=10, fontweight='bold', color='#2c3e50')
    ax_i = fig.add_axes([0.06, y_cur - 0.120, 0.88, 0.112])
    ax_i.axis('off')
    keys = ['ID', 'Nom', 'Altitude', 'Latitude', 'Longitude', 'DateDebut', 'DateFin']
    rows = [[k, str(station_info.iloc[0].get(k, '–'))]
            for k in keys if k in station_info.columns]
    t = ax_i.table(cellText=rows, colLabels=['Parameter', 'Value'],
                   bbox=[0, 0, 1, 1], cellLoc='left')
    _style_table(t)

    # Climatological normals
    y_norm = y_cur - 0.138
    fig.text(0.06, y_norm, 'Climatological Normals (full period)',
             fontsize=10, fontweight='bold', color='#2c3e50')
    ax_n = fig.add_axes([0.06, 0.16, 0.88, y_norm - 0.18])
    ax_n.axis('off')

    _NORM = [
        ('T',     '2 m Temperature',      '°C',    'mean'),
        ('TD',    'Dew Point',            '°C',    'mean'),
        ('U',     'Relative Humidity',    '%',     'mean'),
        ('UABS',  'Absolute Humidity',    'g/m³',  'mean'),
        ('PSTAT', 'Station Pressure',     'hPa',   'mean'),
        ('GLO',   'Global Radiation',     'W/m²',  'mean'),
        ('FF',    'Wind Speed',           'm/s',   'mean'),
        ('RR1',   'Annual Precipitation', 'mm',    'sum'),
        ('N',     'Cloud Cover',          'oktas', 'mean'),
    ]
    norm_rows = []
    for var, label, unit, agg in _NORM:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        if s.empty:
            continue
        ann = _annual_agg(df[var], dti, agg)
        norm_rows.append([
            label, unit,
            f"{ann.mean():.1f}" if agg == 'sum' else f"{s.mean():.2f}",
            f"{s.min():.2f}",
            f"{s.max():.2f}",
            f"{len(s):,}",
            f"{100 * df[var].isna().sum() / max(len(df), 1):.1f}%",
        ])
    tn = ax_n.table(
        cellText=norm_rows,
        colLabels=['Variable', 'Unit', 'Mean', 'Min', 'Max', 'Valid N', 'Missing'],
        bbox=[0, 0, 1, 1], cellLoc='center',
        colWidths=[0.26, 0.08, 0.10, 0.10, 0.10, 0.11, 0.12],
    )
    _style_table(tn)
    for (r, c), cell in tn.get_celld().items():
        if r > 0 and c == 0:
            cell.set_text_props(ha='left')

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_cover.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Temperature page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_temperature(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
                      dti: pd.DatetimeIndex,
                      figures_dir: Optional[str] = None) -> None:
    if 'T' not in df.columns or not df['T'].notna().any():
        return
    T      = df['T']
    has_td = 'TD' in df.columns and df['TD'].notna().any()

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    title = "2 m Air Temperature" + ("  &  Dew Point" if has_td else "")
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.985)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1.8, 1.4],
                          hspace=0.55, wspace=0.38,
                          left=0.09, right=0.95, top=0.958, bottom=0.04)

    # ── Monthly climatology (daily Tmax / Tmean / Tmin) ───────────────────
    ax_c  = fig.add_subplot(gs[0, :])
    T_s   = pd.Series(T.values, index=dti)
    daily = T_s.resample('D').agg(['max', 'min', 'mean'])
    daily['m'] = daily.index.month
    g          = daily.groupby('m')
    mn_tmax    = g['max'].mean()
    mn_tmin    = g['min'].mean()
    mn_tmean   = g['mean'].mean()
    sd_tmax    = g['max'].std()
    sd_tmin    = g['min'].std()
    abs_tmax   = g['max'].max()
    abs_tmin   = g['min'].min()
    x = np.arange(1, 13)

    ax_c.fill_between(x, mn_tmin, mn_tmax, alpha=0.07, color='#c0392b', zorder=1)
    ax_c.fill_between(x, mn_tmax - sd_tmax, mn_tmax + sd_tmax,
                      alpha=0.18, color='#e74c3c', zorder=1)
    ax_c.fill_between(x, mn_tmin - sd_tmin, mn_tmin + sd_tmin,
                      alpha=0.18, color='#2980b9', zorder=1)
    ax_c.plot(x, mn_tmax,  color='#e74c3c', lw=2.0, marker='o', ms=4,
              label='Mean daily max')
    ax_c.plot(x, mn_tmean, color='#2c3e50', lw=2.0, marker='s', ms=3.5,
              ls='--', label='Mean temperature')
    ax_c.plot(x, mn_tmin,  color='#2980b9', lw=2.0, marker='o', ms=4,
              label='Mean daily min')
    ax_c.scatter(x, abs_tmax, marker='^', s=55, color='#922b21',
                 zorder=5, label='Monthly record max')
    ax_c.scatter(x, abs_tmin, marker='v', s=55, color='#1a5276',
                 zorder=5, label='Monthly record min')
    if has_td:
        TD_s  = pd.Series(df['TD'].values, index=dti)
        mn_td = TD_s.resample('D').mean().to_frame('v')
        mn_td['m'] = mn_td.index.month
        mn_td_m = mn_td.groupby('m')['v'].mean()
        ax_c.plot(x, mn_td_m.values, color='#27ae60', lw=1.6, marker='D',
                  ms=3, ls=':', label='Mean dew point', alpha=0.85)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_MONTH_ABBR, fontsize=8)
    ax_c.set_ylabel('Temperature (°C)', fontsize=9)
    ax_c.legend(fontsize=6.5, ncol=3, loc='best', framealpha=0.8)
    ax_c.grid(axis='y', alpha=0.20)
    ax_c.tick_params(axis='y', labelsize=7)
    ax_c.set_title(
        'Monthly Climatology  (shading = ±1 σ interannual  |  triangles = absolute records)',
        fontsize=7.5, loc='left', pad=3)

    # ── Diurnal cycle by season ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    _diurnal_ax(ax_d, T, dti, '°C')
    ax_d.set_title('Diurnal Cycle by Season', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Annual mean + Theil-Sen trend ─────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 1])
    ann  = _annual_agg(T, dti, 'mean')
    tau, p, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p, slope, intercept, '°C')
    _mk_box(ax_t, _mk_text(tau, p, slope, '°C'), p)
    ax_t.set_title('Annual Mean Temperature + Trend', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Records table ─────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[2, :])
    ax_r.set_title('Temperature Records — Top 5', fontsize=9,
                   fontweight='bold', loc='left', pad=3)
    _records_panel(ax_r, _top5(T, dates, 'max'), _top5(T, dates, 'min'),
                   'Max T (°C)', 'Min T (°C)', '°C')

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_temperature.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Humidity page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_humidity(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
                   dti: pd.DatetimeIndex,
                   figures_dir: Optional[str] = None) -> None:
    has_u    = 'U'    in df.columns and df['U'].notna().any()
    has_uabs = 'UABS' in df.columns and df['UABS'].notna().any()
    if not has_u and not has_uabs:
        return

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle("Relative & Absolute Humidity", fontsize=14,
                 fontweight='bold', y=0.985)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1.8, 1.4],
                          hspace=0.55, wspace=0.38,
                          left=0.09, right=0.95, top=0.958, bottom=0.04)

    # ── Monthly climatology ───────────────────────────────────────────────
    ax_c  = fig.add_subplot(gs[0, :])
    ax_c2 = ax_c.twinx() if (has_u and has_uabs) else None
    x     = np.arange(1, 13)

    if has_u:
        g  = _grp_month(df['U'], dti)
        mu, sd = g.mean(), g.std()
        ax_c.plot(x, mu.values, color='#2980b9', lw=2, marker='o', ms=4,
                  label='Relative Humidity (%)')
        ax_c.fill_between(x, mu - sd, mu + sd, alpha=0.15, color='#2980b9')
        ax_c.set_ylabel('Relative Humidity (%)', fontsize=8)
        ax_c.tick_params(axis='y', labelsize=7)

    if has_uabs:
        target = ax_c2 if ax_c2 is not None else ax_c
        g2     = _grp_month(df['UABS'], dti)
        mu2, sd2 = g2.mean(), g2.std()
        target.plot(x, mu2.values, color='#27ae60', lw=2, marker='s', ms=4,
                    ls='--', label='Absolute Humidity (g/m³)')
        target.fill_between(x, mu2 - sd2, mu2 + sd2, alpha=0.12, color='#27ae60')
        target.set_ylabel('Absolute Humidity (g/m³)', fontsize=8)
        target.tick_params(axis='y', labelsize=7)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_MONTH_ABBR, fontsize=8)
    ax_c.grid(axis='y', alpha=0.20)
    lines1, lbl1 = ax_c.get_legend_handles_labels()
    lines2, lbl2 = (ax_c2.get_legend_handles_labels() if ax_c2 else ([], []))
    ax_c.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, framealpha=0.8)
    ax_c.set_title('Monthly Climatology (mean ± 1 σ)', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Diurnal cycle ─────────────────────────────────────────────────────
    primary = df['U'] if has_u else df['UABS']
    unit_p  = '%' if has_u else 'g/m³'
    ax_d = fig.add_subplot(gs[1, 0])
    _diurnal_ax(ax_d, primary, dti, unit_p)
    lbl = 'Relative Humidity' if has_u else 'Absolute Humidity'
    ax_d.set_title(f'Diurnal Cycle — {lbl}', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Annual trend ──────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 1])
    ann  = _annual_agg(primary, dti, 'mean')
    tau, p, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p, slope, intercept, unit_p)
    _mk_box(ax_t, _mk_text(tau, p, slope, unit_p), p)
    ax_t.set_title('Annual Mean + Trend', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Records ───────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[2, :])
    ax_r.set_title('Humidity Records — Top 5', fontsize=9,
                   fontweight='bold', loc='left', pad=3)
    _records_panel(ax_r, _top5(primary, dates, 'max'), _top5(primary, dates, 'min'),
                   f'Max ({unit_p})', f'Min ({unit_p})', unit_p)

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_humidity.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Pressure page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_pressure(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
                   dti: pd.DatetimeIndex,
                   figures_dir: Optional[str] = None) -> None:
    if 'PSTAT' not in df.columns or not df['PSTAT'].notna().any():
        return
    P = df['PSTAT']

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle("Station Pressure", fontsize=14, fontweight='bold', y=0.985)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1.8, 1.4],
                          hspace=0.55, wspace=0.38,
                          left=0.09, right=0.95, top=0.958, bottom=0.04)

    # ── Monthly climatology ───────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, :])
    g    = _grp_month(P, dti)
    mu, sd = g.mean(), g.std()
    p05, p95 = g.quantile(0.05), g.quantile(0.95)
    x = np.arange(1, 13)
    ax_c.fill_between(x, p05.values, p95.values, alpha=0.10,
                      color='#8e44ad', label='5th–95th percentile')
    ax_c.fill_between(x, mu - sd, mu + sd, alpha=0.20,
                      color='#8e44ad', label='±1 σ')
    ax_c.plot(x, mu.values, color='#8e44ad', lw=2, marker='o', ms=4, label='Mean')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_MONTH_ABBR, fontsize=8)
    ax_c.set_ylabel('Pressure (hPa)', fontsize=9)
    ax_c.tick_params(axis='y', labelsize=7)
    ax_c.legend(fontsize=7, framealpha=0.8)
    ax_c.grid(axis='y', alpha=0.20)
    ax_c.set_title('Monthly Climatology (mean ± 1 σ  +  5–95th percentile)',
                   fontsize=8, loc='left', pad=3)

    # ── Diurnal cycle ─────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    _diurnal_ax(ax_d, P, dti, 'hPa')
    ax_d.set_title('Diurnal Cycle by Season', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Annual trend ──────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 1])
    ann  = _annual_agg(P, dti, 'mean')
    tau, p_mk, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p_mk, slope, intercept, 'hPa')
    _mk_box(ax_t, _mk_text(tau, p_mk, slope, 'hPa'), p_mk)
    ax_t.set_title('Annual Mean Pressure + Trend', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Records ───────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[2, :])
    ax_r.set_title('Pressure Records — Top 5', fontsize=9,
                   fontweight='bold', loc='left', pad=3)
    _records_panel(ax_r, _top5(P, dates, 'max'), _top5(P, dates, 'min'),
                   'Max PSTAT (hPa)', 'Min PSTAT (hPa)', 'hPa')

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_pressure.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Wind page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_wind(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
               dti: pd.DatetimeIndex,
               figures_dir: Optional[str] = None) -> None:
    has_ff = 'FF' in df.columns and df['FF'].notna().any()
    has_dd = 'DD' in df.columns and df['DD'].notna().any()
    if not has_ff:
        return
    FF = df['FF']
    DD = df['DD'] if has_dd else pd.Series(np.nan, index=FF.index)

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle("Wind Speed & Direction", fontsize=14, fontweight='bold', y=0.985)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1.8, 1.4],
                          hspace=0.55, wspace=0.42,
                          left=0.07, right=0.95, top=0.958, bottom=0.04)

    # ── Wind rose ─────────────────────────────────────────────────────────
    ax_rose = fig.add_subplot(gs[0, 0], projection='polar')
    _wind_rose_ax(ax_rose, FF, DD)
    legend_handles = [Patch(facecolor=c, label=f'{l} m/s')
                      for c, l in zip(_SPD_COLORS, _SPD_LABELS)]
    ax_rose.legend(handles=legend_handles, fontsize=6, loc='lower center',
                   bbox_to_anchor=(0.5, -0.24), ncol=3, framealpha=0.8)

    # ── Monthly FF boxplot ────────────────────────────────────────────────
    ax_box = fig.add_subplot(gs[0, 1])
    monthly_ff = [
        pd.Series(FF.values, index=dti)[dti.month == m].dropna().values
        for m in range(1, 13)
    ]
    bp = ax_box.boxplot(monthly_ff, patch_artist=True, widths=0.6,
                        medianprops=dict(color='white', lw=1.5),
                        flierprops=dict(marker='.', ms=2, alpha=0.3))
    for patch in bp['boxes']:
        patch.set_facecolor('#2980b9')
        patch.set_alpha(0.7)
    ax_box.set_xticks(range(1, 13))
    ax_box.set_xticklabels(_MONTH_ABBR, fontsize=7)
    ax_box.tick_params(axis='y', labelsize=7)
    ax_box.set_ylabel('Wind Speed (m/s)', fontsize=8)
    ax_box.grid(axis='y', alpha=0.20)
    ax_box.set_title('Monthly Wind Speed Distribution', fontsize=9,
                     fontweight='bold', loc='left', pad=3)

    # ── Annual mean FF + trend ────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, :])
    ann  = _annual_agg(FF, dti, 'mean')
    tau, p_mk, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p_mk, slope, intercept, 'm/s')
    _mk_box(ax_t, _mk_text(tau, p_mk, slope, 'm/s'), p_mk)
    ax_t.set_title('Annual Mean Wind Speed + Trend', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    # ── Records (top 5 strongest gusts with direction) ────────────────────
    ax_r = fig.add_subplot(gs[2, :])
    ax_r.axis('off')
    ax_r.set_title('Wind Records — Top 5 Strongest Hourly Observations',
                   fontsize=9, fontweight='bold', loc='left', pad=3)
    top_idx = FF.nlargest(5).index
    rec_rows = []
    for rank, idx in enumerate(top_idx, 1):
        ff_val  = FF.loc[idx]
        dd_val  = DD.loc[idx]
        dir_str = f"{dd_val:.0f}°" if not np.isnan(dd_val) else '–'
        date_str = dates.loc[idx].strftime('%Y-%m-%d %Hh')
        rec_rows.append([str(rank), f"{ff_val:.1f}", dir_str, date_str])
    if rec_rows:
        t = ax_r.table(cellText=rec_rows,
                       colLabels=['#', 'FF (m/s)', 'DD (°)', 'Date'],
                       bbox=[0.08, 0, 0.84, 1], cellLoc='center')
        _style_table(t, header_color='#2c3e50')

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_wind.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Radiation page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_radiation(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
                    dti: pd.DatetimeIndex,
                    figures_dir: Optional[str] = None) -> None:
    rad_vars = {v: l for v, l in [('GLO', 'Global'), ('DIR', 'Direct'),
                                   ('DIF', 'Diffuse'), ('INFRAR', 'Infrared')]
                if v in df.columns and df[v].notna().any()}
    if not rad_vars:
        return

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle("Solar & Thermal Radiation", fontsize=14, fontweight='bold', y=0.985)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 2.0, 1.6],
                          hspace=0.50,
                          left=0.09, right=0.93, top=0.958, bottom=0.05)

    # ── Monthly climatology: mean + p90 ───────────────────────────────────
    ax_c = fig.add_subplot(gs[0])
    _RAD_COLORS = {'GLO': '#e67e22', 'DIR': '#f39c12',
                   'DIF': '#3498db', 'INFRAR': '#9b59b6'}
    x = np.arange(1, 13)
    for var, label in rad_vars.items():
        g   = _grp_month(df[var], dti)
        mu  = g.mean()
        p90 = g.quantile(0.90)
        c   = _RAD_COLORS[var]
        ax_c.plot(x, mu.values,  color=c, lw=2,   marker='o', ms=3.5,
                  label=f'{label} – mean')
        ax_c.plot(x, p90.values, color=c, lw=1.2, ls=':',     alpha=0.65,
                  label=f'{label} – p90')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_MONTH_ABBR, fontsize=8)
    ax_c.set_ylabel('Radiation (W/m²)', fontsize=9)
    ax_c.tick_params(axis='y', labelsize=7)
    ax_c.legend(fontsize=6.5, ncol=2, framealpha=0.8)
    ax_c.grid(axis='y', alpha=0.20)
    ax_c.set_title('Monthly Climatology (mean — p90 dotted)',
                   fontsize=8, loc='left', pad=3)

    # ── Diurnal heatmap (month × hour) ────────────────────────────────────
    ax_hm  = fig.add_subplot(gs[1])
    best   = next(v for v in ['GLO', 'DIR', 'DIF', 'INFRAR'] if v in rad_vars)
    months = np.array(dti.month)
    hours  = np.array(dti.hour)
    tmp_hm = pd.DataFrame({'v': df[best].values, 'm': months, 'h': hours})
    pivot  = tmp_hm.groupby(['m', 'h'])['v'].mean().unstack(fill_value=0)
    # Ensure all 24 hours are present
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0.0
    pivot = pivot[sorted(pivot.columns)]
    im = ax_hm.pcolormesh(np.arange(24), np.arange(1, 13), pivot.values,
                          cmap='YlOrRd', shading='auto')
    cbar = plt.colorbar(im, ax=ax_hm, label='W/m²', fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax_hm.set_xticks(range(0, 24, 2))
    ax_hm.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 2)], fontsize=7)
    ax_hm.set_yticks(range(1, 13))
    ax_hm.set_yticklabels(_MONTH_ABBR, fontsize=7)
    ax_hm.set_xlabel('Hour (UTC)', fontsize=8)
    ax_hm.set_title(
        f'Mean Diurnal Cycle by Month — {rad_vars[best]} Radiation (W/m²)',
        fontsize=9, fontweight='bold', loc='left', pad=3)

    # ── Annual mean trend ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[2])
    ann  = _annual_agg(df[best], dti, 'mean')
    tau, p_mk, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p_mk, slope, intercept, 'W/m²')
    _mk_box(ax_t, _mk_text(tau, p_mk, slope, 'W/m²'), p_mk)
    ax_t.set_title(f'Annual Mean {rad_vars[best]} Radiation + Trend',
                   fontsize=9, fontweight='bold', loc='left', pad=3)

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_radiation.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Precipitation page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_precipitation(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
                        dti: pd.DatetimeIndex,
                        figures_dir: Optional[str] = None) -> None:
    if 'RR1' not in df.columns or not df['RR1'].notna().any():
        return
    RR = df['RR1']

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle("Precipitation", fontsize=14, fontweight='bold', y=0.985)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.8, 1.4],
                          hspace=0.55,
                          left=0.10, right=0.95, top=0.958, bottom=0.05)

    # ── Monthly climatological cumul ──────────────────────────────────────
    ax_c  = fig.add_subplot(gs[0])
    RR_s  = pd.Series(RR.values, index=dti)
    # Monthly totals per year, then average per calendar month
    mon_tot    = RR_s.resample('ME').sum(min_count=1)
    mon_tot_df = pd.DataFrame({'v': mon_tot.values, 'm': mon_tot.index.month})
    clim  = mon_tot_df.groupby('m')['v']
    mu_m  = clim.mean()
    sd_m  = clim.std()
    p75_m = clim.quantile(0.75)
    x     = np.arange(1, 13)
    bars  = ax_c.bar(x, mu_m.values, color='#2980b9', edgecolor='white',
                     linewidth=0.5, alpha=0.85, label='Mean monthly total')
    ax_c.errorbar(x, mu_m.values, yerr=sd_m.values, fmt='none',
                  color='#1a5276', capsize=4, lw=1.2, label='±1 σ')
    ax_c.plot(x, p75_m.values, color='#e74c3c', lw=1.5, ls='--',
              marker='o', ms=3.5, label='75th percentile')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_MONTH_ABBR, fontsize=8)
    ax_c.set_ylabel('Precipitation (mm/month)', fontsize=9)
    ax_c.tick_params(axis='y', labelsize=7)
    ax_c.legend(fontsize=7, framealpha=0.8)
    ax_c.grid(axis='y', alpha=0.20)
    ax_c.set_title('Monthly Climatological Precipitation (mean ± 1 σ)',
                   fontsize=8, loc='left', pad=3)

    # ── Annual total + trend ──────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1])
    ann  = _annual_agg(RR, dti, 'sum')
    tau, p_mk, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p_mk, slope, intercept, 'mm')
    _mk_box(ax_t, _mk_text(tau, p_mk, slope, 'mm'), p_mk)
    ax_t.set_title('Annual Total Precipitation + Trend',
                   fontsize=9, fontweight='bold', loc='left', pad=3)

    # ── Extreme events table ──────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[2])
    ax_r.axis('off')
    ax_r.set_title('Top 5 Hourly Rainfall Events', fontsize=9,
                   fontweight='bold', loc='left', pad=3)
    rec  = _top5(RR, dates, 'max')
    rows = [[str(r.Rank), f"{r.Value:.1f} mm", r.Date]
            for r in rec.itertuples()] if not rec.empty else []
    if rows:
        t = ax_r.table(cellText=rows,
                       colLabels=['#', 'Hourly RR (mm)', 'Date'],
                       bbox=[0.10, 0, 0.80, 1], cellLoc='center')
        _style_table(t, header_color='#2980b9')

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_precipitation.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Cloud cover page
# ═══════════════════════════════════════════════════════════════════════════════

def _page_cloudcover(pdf: PdfPages, df: pd.DataFrame, dates: pd.Series,
                     dti: pd.DatetimeIndex,
                     figures_dir: Optional[str] = None) -> None:
    if 'N' not in df.columns or not df['N'].notna().any():
        return
    N = df['N']

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle("Cloud Cover", fontsize=14, fontweight='bold', y=0.985)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.8],
                          hspace=0.50,
                          left=0.09, right=0.95, top=0.958, bottom=0.06)

    # ── Monthly okta frequency (stacked bar) ──────────────────────────────
    ax_c = fig.add_subplot(gs[0])
    N_arr   = np.round(N.values).astype(float)
    mon_arr = np.array(dti.month)
    df_n    = pd.DataFrame({'N': N_arr, 'm': mon_arr})
    df_n    = df_n.dropna()
    df_n['N'] = df_n['N'].astype(int).clip(0, 8)
    pivot     = df_n.groupby(['m', 'N']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    try:
        cmap_n = plt.cm.get_cmap('RdYlBu_r', 9)
    except Exception:
        cmap_n = plt.colormaps.get_cmap('RdYlBu_r').resampled(9)

    x       = np.arange(1, 13)
    bottom_ = np.zeros(12)
    for octa in range(9):
        if octa not in pivot_pct.columns:
            continue
        vals_ = np.array([pivot_pct.loc[m, octa] if m in pivot_pct.index else 0.0
                          for m in x])
        lbl = f'{octa} okta'
        if octa == 0:
            lbl += ' (clear)'
        elif octa == 8:
            lbl += ' (overcast)'
        ax_c.bar(x, vals_, bottom=bottom_, color=cmap_n(octa / 8),
                 edgecolor='white', linewidth=0.3, label=lbl)
        bottom_ += vals_

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_MONTH_ABBR, fontsize=8)
    ax_c.set_ylabel('Frequency (%)', fontsize=9)
    ax_c.set_ylim(0, 100)
    ax_c.tick_params(axis='y', labelsize=7)
    ax_c.legend(fontsize=6, ncol=5, loc='upper right', framealpha=0.8)
    ax_c.grid(axis='y', alpha=0.15)
    ax_c.set_title('Monthly Cloud Cover Frequency by Okta Class',
                   fontsize=9, fontweight='bold', loc='left', pad=3)

    # ── Annual mean + trend ───────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1])
    ann  = _annual_agg(N, dti, 'mean')
    tau, p_mk, slope, intercept = _mann_kendall(ann.values)
    _trend_ax(ax_t, ann, tau, p_mk, slope, intercept, 'oktas')
    _mk_box(ax_t, _mk_text(tau, p_mk, slope, 'oktas'), p_mk)
    ax_t.set_title('Annual Mean Cloud Cover + Trend', fontsize=9,
                   fontweight='bold', loc='left', pad=3)

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'factsheet_cloudcover.png'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_factsheet_pdf(dataset_manager, station_info: pd.DataFrame,
                           output_dir: str = "out") -> str:
    """
    Generate a climatological factsheet PDF for the station.
    One page per variable group (cover + 7 variable pages).
    Skips pages whose variables are absent from the dataset.
    Returns the path to the generated PDF, or empty string on failure.
    """
    df    = dataset_manager.data
    dates = df['DATE']
    dti   = pd.DatetimeIndex(dates)

    n_years = (dti.max() - dti.min()).days / 365.25
    if n_years < MIN_YEARS_WARNING:
        logger.warning(
            "Factsheet — only %.0f years of data. "
            "Climatological statistics may not be representative.", n_years)

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'factsheet.pdf')

    with PdfPages(out_path) as pdf:
        _cover_page(pdf, df, station_info, n_years, dti, figures_dir)
        _page_temperature(pdf, df, dates, dti, figures_dir)
        _page_humidity(pdf, df, dates, dti, figures_dir)
        _page_pressure(pdf, df, dates, dti, figures_dir)
        _page_wind(pdf, df, dates, dti, figures_dir)
        _page_radiation(pdf, df, dates, dti, figures_dir)
        _page_precipitation(pdf, df, dates, dti, figures_dir)
        _page_cloudcover(pdf, df, dates, dti, figures_dir)

    logger.info("Factsheet saved: %s", out_path)
    return out_path
