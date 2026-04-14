"""
Quality control report generator - MeteoFrance hourly data.
Outputs quality_checks.pdf in the specified output directory.
Individual figures are also saved as PNG in output_dir/figures/.

Checks per variable:
  - Missing values
  - Physical range (ABSURD)
  - Temporal jumps (JUMP)
  - Constant plateaus (PLATEAU)
  - Contextual outliers by (month, hour) via 3xIQR (CONTEXTUAL)

Global checks (cover page):
  - Temporal continuity (gaps in the time index)
  - Duplicate timestamps
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# ── Variable configurations ────────────────────────────────────────────────

@dataclass
class VarConfig:
    label:      str
    unit:       str
    phys_min:   Optional[float]
    phys_max:   Optional[float]
    delta_max:  Optional[float] = None   # max acceptable hourly change
    plateau_h:  Optional[int]   = 24     # consecutive equal values (hours); None = skip

VARIABLES: dict = {
    'T':      VarConfig('2 m Air Temperature',      '°C',     -60,   60,  delta_max=8,  plateau_h=24),
    'TD':     VarConfig('Dew Point Temperature',    '°C',     -80,   50,  delta_max=10,  plateau_h=24),
    'U':      VarConfig('Relative Humidity',        '%',        0,  100,  delta_max=None, plateau_h=24),
    'UABS':   VarConfig('Absolute Humidity',        'g/m³',     0,   50,  delta_max=None, plateau_h=None),
    'PSTAT':  VarConfig('Station Pressure',         'hPa',    870, 1084,  delta_max=5,   plateau_h=24),
    'GLO':    VarConfig('Global Radiation',         'W/m²',   -10, 1400,  delta_max=None, plateau_h=None),
    'DIR':    VarConfig('Direct Radiation',         'W/m²',   -10, 1400,  delta_max=None, plateau_h=None),
    'DIF':    VarConfig('Diffuse Radiation',        'W/m²',   -10,  800,  delta_max=None, plateau_h=None),
    'INFRAR': VarConfig('Infrared Radiation',       'W/m²',   150,  700,  delta_max=None, plateau_h=None),
    'N':      VarConfig('Cloud Cover',              'octas',    0,    8,  delta_max=None, plateau_h=None),
    'DD':     VarConfig('Wind Direction',           '°',        0,  360,  delta_max=None, plateau_h=None),
    'FF':     VarConfig('Wind Speed',               'm/s',      0,   75,  delta_max=20,  plateau_h=12),
    'RR1':    VarConfig('Hourly Precipitation',     'mm',       0,  200,  delta_max=None, plateau_h=None),
}

_FLAG_COLORS = {
    'ABSURD':      '#d62728',
    'JUMP':        '#ff7f0e',
    'PLATEAU':     '#1f77b4',
    'CONTEXTUAL':  '#9467bd',
}

_FLAG_BG = {
    'MISSING':    '#f5f5f5',
    'ABSURD':     '#ffd7d7',
    'JUMP':       '#ffe8cc',
    'PLATEAU':    '#d7e8ff',
    'CONTEXTUAL': '#ead7ff',
    'TOTAL':      '#fff3b0',
}

# ── Core check per variable ────────────────────────────────────────────────

def _check_variable(series: pd.Series, dates: pd.Series, cfg: VarConfig) -> dict:
    n = len(series)
    mask_missing = series.isna()
    n_missing = int(mask_missing.sum())

    # Physical range
    if cfg.phys_min is not None and cfg.phys_max is not None:
        mask_absurd = ((series < cfg.phys_min) | (series > cfg.phys_max)) & ~mask_missing
    else:
        mask_absurd = pd.Series(False, index=series.index)

    # Temporal jumps
    if cfg.delta_max is not None:
        mask_jump = (series.diff().abs() > cfg.delta_max) & ~mask_missing
    else:
        mask_jump = pd.Series(False, index=series.index)

    # Constant plateaus
    if cfg.plateau_h is not None:
        runs        = (series.diff().ne(0) | mask_missing).cumsum()
        run_lengths = series.groupby(runs).transform('count')
        mask_flat   = (run_lengths >= cfg.plateau_h) & ~mask_missing
    else:
        mask_flat = pd.Series(False, index=series.index)

    # Contextual outliers – Q1−3×IQR / Q3+3×IQR by (month, hour)
    try:
        month = dates.dt.month.values
        hour  = dates.dt.hour.values
        tmp   = pd.DataFrame({'v': series.values, 'm': month, 'h': hour})
        qs    = tmp.groupby(['m', 'h'])['v'].quantile([0.25, 0.75]).unstack()
        qs.columns = ['q25', 'q75']
        qs['iqr']   = qs['q75'] - qs['q25']
        qs['lower'] = qs['q25'] - 3 * qs['iqr']
        qs['upper'] = qs['q75'] + 3 * qs['iqr']
        mh_idx = pd.MultiIndex.from_arrays([month, hour])
        lo     = qs['lower'].reindex(mh_idx).values
        hi     = qs['upper'].reindex(mh_idx).values
        mask_ctx = pd.Series(
            (series.values < lo) | (series.values > hi),
            index=series.index,
        ) & ~mask_missing & ~mask_absurd
    except Exception:
        mask_ctx = pd.Series(False, index=series.index)

    # Combine flags
    flag_col = pd.Series('', index=series.index, dtype=object)
    for mask, code in [
        (mask_missing, 'MISSING'),
        (mask_absurd,  'ABSURD'),
        (mask_jump,    'JUMP'),
        (mask_flat,    'PLATEAU'),
        (mask_ctx,     'CONTEXTUAL'),
    ]:
        empty  = mask & (flag_col == '')
        filled = mask & (flag_col != '')
        flag_col[empty]  = code
        flag_col[filled] = flag_col[filled] + '|' + code

    return {
        'n':            n,
        'n_missing':    n_missing,
        'pct_missing':  100 * n_missing / n if n > 0 else 0.0,
        'n_absurd':     int(mask_absurd.sum()),
        'n_jumps':      int(mask_jump.sum()),
        'n_plateau':    int(mask_flat.sum()),
        'n_contextual': int(mask_ctx.sum()),
        'flags':        flag_col,
        'mask_missing': mask_missing,
        'mask_absurd':  mask_absurd,
        'mask_jump':    mask_jump,
        'mask_flat':    mask_flat,
        'mask_ctx':     mask_ctx,
    }


# ── Table style helper ─────────────────────────────────────────────────────

def _style_table(tbl, header_color='#2c3e50', stripe_color='#ecf0f1'):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor(stripe_color)


# ── Cover page ─────────────────────────────────────────────────────────────

def _cover_page(pdf: PdfPages, df: pd.DataFrame, station_info: pd.DataFrame,
                all_checks: dict, n_gaps: int, n_dup: int,
                figures_dir: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # ── Header ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975, "Quality Control Report",
             ha='center', fontsize=18, fontweight='bold')
    station_label = (f"{station_info.iloc[0].get('ID', '')}  –  "
                     f"{station_info.iloc[0].get('Nom', '')}")
    fig.text(0.5, 0.951, station_label, ha='center', fontsize=13, color='#333333')
    period = (f"{df['DATE'].min().strftime('%d/%m/%Y')} "
              f"→  {df['DATE'].max().strftime('%d/%m/%Y')}")
    fig.text(0.5, 0.930, period, ha='center', fontsize=11, color='#555555')
    fig.text(0.5, 0.912, f"Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}",
             ha='center', fontsize=9, color='#777777')

    # ── Station info ──────────────────────────────────────────────────────
    fig.text(0.06, 0.893, 'Station Information',
             fontsize=10, fontweight='bold', color='#2c3e50')
    ax_info = fig.add_axes([0.06, 0.768, 0.88, 0.118])
    ax_info.axis('off')
    info_keys = ['ID', 'Nom', 'Altitude', 'Latitude', 'Longitude', 'DateDebut', 'DateFin']
    info_rows = [[k, str(station_info.iloc[0].get(k, '–'))]
                 for k in info_keys if k in station_info.columns]
    tbl = ax_info.table(cellText=info_rows, colLabels=['Parameter', 'Value'],
                        bbox=[0, 0, 1, 1], cellLoc='left')
    _style_table(tbl)

    # ── Global dataset info ───────────────────────────────────────────────
    fig.text(0.06, 0.750, 'Time Series',
             fontsize=10, fontweight='bold', color='#2c3e50')
    ax_glob = fig.add_axes([0.06, 0.662, 0.88, 0.080])
    ax_glob.axis('off')
    dt_mode = df['DATE'].diff().dropna().mode()
    dt_str  = str(dt_mode.iloc[0]) if len(dt_mode) else '–'
    glob_rows = [
        ['Detected time step',              dt_str],
        ['Number of time steps',            f"{len(df):,}"],
        ['Temporal gaps (irregular steps)', str(n_gaps)],
        ['Duplicate timestamps',            str(n_dup)],
    ]
    tbl2 = ax_glob.table(cellText=glob_rows, colLabels=['Indicator', 'Value'],
                         bbox=[0, 0, 1, 1], cellLoc='left')
    _style_table(tbl2)

    # ── Quality flag descriptions ─────────────────────────────────────────
    fig.text(0.06, 0.644, 'Quality Flags',
             fontsize=10, fontweight='bold', color='#2c3e50')
    ax_flags = fig.add_axes([0.06, 0.548, 0.88, 0.088])
    ax_flags.axis('off')
    flag_rows = [
        ['MISSING',    'Timestamp present but no measurement recorded.'],
        ['ABSURD',     'Value outside physical bounds (variable-specific thresholds — see documentation*).'],
        ['JUMP',       'Hourly change exceeds the maximum allowed delta (variable-specific — see documentation*).'],
        ['PLATEAU',    'Too many consecutive identical values (variable-specific minimum run length — see documentation*).'],
        ['CONTEXTUAL', 'Statistical outlier vs. typical (month, hour) distribution  [Q1 − 3×IQR,  Q3 + 3×IQR].'],
    ]
    tbl_flags = ax_flags.table(cellText=flag_rows, colLabels=['Flag', 'Description'],
                               bbox=[0, 0, 1, 1], cellLoc='left',
                               colWidths=[0.13, 0.87])
    tbl_flags.auto_set_font_size(False)
    tbl_flags.set_fontsize(7.5)
    tbl_flags.scale(1, 1.30)
    for (r, c), cell in tbl_flags.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            flag_code = flag_rows[r - 1][0] if r <= len(flag_rows) else ''
            cell.set_facecolor(_FLAG_BG.get(flag_code, '#ffffff'))
    fig.text(0.06, 0.540,
             '* Per-variable threshold documentation — coming soon.',
             fontsize=7, color='#888888', style='italic')

    # ── Summary table (all variables) ─────────────────────────────────────
    fig.text(0.06, 0.526, 'Quality Summary by Variable',
             fontsize=10, fontweight='bold', color='#2c3e50')
    ax_sum = fig.add_axes([0.06, 0.035, 0.88, 0.483])
    ax_sum.axis('off')
    col_labels = ['Variable', 'Unit', 'N', 'Missing', 'Absurd',
                  'Jumps', 'Plateaus', 'Contextual', 'Flagged']
    col_widths = [0.24, 0.07, 0.08, 0.155, 0.07, 0.07, 0.08, 0.095, 0.08]
    rows = []
    for var, chk in all_checks.items():
        cfg = VARIABLES.get(var)
        n_flagged   = int((chk['flags'] != '').sum())
        pct_flagged = 100 * n_flagged / chk['n'] if chk['n'] > 0 else 0
        rows.append([
            cfg.label if cfg else var,
            cfg.unit  if cfg else '',
            f"{chk['n']:,}",
            f"{chk['n_missing']:,}\n({chk['pct_missing']:.1f} %)",
            str(chk['n_absurd']),
            str(chk['n_jumps']),
            str(chk['n_plateau']),
            str(chk['n_contextual']),
            f"{pct_flagged:.1f} %",
        ])
    tbl3 = ax_sum.table(cellText=rows, colLabels=col_labels,
                        bbox=[0, 0, 1, 1], cellLoc='center',
                        colWidths=col_widths)
    _style_table(tbl3)
    for (r, c), cell in tbl3.get_celld().items():
        if r > 0 and c == 0:
            cell.set_text_props(ha='left')

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, 'cover.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


# ── Per-variable page ──────────────────────────────────────────────────────

def _variable_page(pdf: PdfPages, var: str, series: pd.Series,
                   dates: pd.Series, cfg: VarConfig, chk: dict,
                   figures_dir: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.suptitle(f"{cfg.label}  ({cfg.unit})", fontsize=14, fontweight='bold', y=0.985)

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[2.8, 1.4, 1.8],
        hspace=0.50, wspace=0.35,
        left=0.09, right=0.95, top=0.955, bottom=0.05,
    )

    # ── Time series with flags ────────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[0, :])
    clean = chk['flags'] == ''
    ax_ts.plot(dates[clean], series[clean],
               color='#2c3e50', linewidth=0.35, alpha=0.7, zorder=2,
               label='Valid')
    for code, color in _FLAG_COLORS.items():
        mask = chk['flags'].str.contains(code, regex=False)
        if mask.any():
            ax_ts.scatter(dates[mask], series[mask],
                          color=color, s=7, zorder=3, label=code, alpha=0.85)
    if cfg.phys_min is not None:
        ax_ts.axhline(cfg.phys_min, color='red', linewidth=0.7, linestyle='--',
                      alpha=0.45, label=f'Physical bounds [{cfg.phys_min}, {cfg.phys_max}]')
        ax_ts.axhline(cfg.phys_max, color='red', linewidth=0.7, linestyle='--', alpha=0.45)
    ax_ts.set_ylabel(f"{cfg.label} ({cfg.unit})", fontsize=9)
    ax_ts.legend(fontsize=7, ncol=4, loc='upper right',
                 framealpha=0.8, markerscale=1.4)
    ax_ts.grid(axis='y', alpha=0.25)
    ax_ts.tick_params(labelsize=8)

    # ── Descriptive statistics ────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis('off')
    valid = series.dropna()
    if len(valid) > 0:
        stats_data = [
            ['Total N',    f"{chk['n']:,}"],
            ['Valid N',    f"{len(valid):,}"],
            ['Missing',    f"{chk['n_missing']:,}  ({chk['pct_missing']:.1f} %)"],
            ['Minimum',    f"{valid.min():.3f}"],
            ['Maximum',    f"{valid.max():.3f}"],
            ['Mean',       f"{valid.mean():.3f}"],
            ['Std Dev',    f"{valid.std():.3f}"],
            ['Median',     f"{valid.median():.3f}"],
            ['Q 0.01',     f"{valid.quantile(0.01):.3f}"],
            ['Q 0.99',     f"{valid.quantile(0.99):.3f}"],
        ]
    else:
        stats_data = [['Total N', f"{chk['n']:,}"], ['Data', 'All missing']]
    tbl = ax_tbl.table(cellText=stats_data, colLabels=['Statistic', 'Value'],
                       loc='upper center', cellLoc='left')
    _style_table(tbl)
    ax_tbl.set_title('Descriptive Statistics', fontsize=9,
                     fontweight='bold', loc='left', pad=4)

    # ── Flags summary ─────────────────────────────────────────────────────
    ax_flg = fig.add_subplot(gs[1, 1])
    ax_flg.axis('off')
    n = chk['n']

    def _pct(k):
        return f"{100 * chk[k] / n:.2f} %" if n else '–'

    n_flagged   = int((chk['flags'] != '').sum())
    pct_flagged = f"{100 * n_flagged / n:.2f} %" if n else '–'
    flg_rows = [
        ['MISSING',    f"{chk['n_missing']:,}",    _pct('n_missing')],
        ['ABSURD',     f"{chk['n_absurd']:,}",     _pct('n_absurd')],
        ['JUMP',       f"{chk['n_jumps']:,}",      _pct('n_jumps')],
        ['PLATEAU',    f"{chk['n_plateau']:,}",    _pct('n_plateau')],
        ['CONTEXTUAL', f"{chk['n_contextual']:,}", _pct('n_contextual')],
        ['TOTAL',      f"{n_flagged:,}",           pct_flagged],
    ]
    tbl2 = ax_flg.table(cellText=flg_rows, colLabels=['Flag', 'Count', '%'],
                        loc='upper center', cellLoc='center')
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8)
    tbl2.scale(1, 1.35)
    code_order = ['MISSING', 'ABSURD', 'JUMP', 'PLATEAU', 'CONTEXTUAL', 'TOTAL']
    for (r, c), cell in tbl2.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif 1 <= r <= len(code_order):
            bg = _FLAG_BG.get(code_order[r - 1], '#ffffff')
            cell.set_facecolor(bg)
    ax_flg.set_title('Quality Flags', fontsize=9, fontweight='bold', loc='left', pad=4)

    # ── Missing values per year ───────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[2, :])
    year   = dates.dt.year
    na_yr  = series.isna().groupby(year).sum()
    tot_yr = series.groupby(year).size()
    pct_yr = 100 * na_yr / tot_yr

    if len(na_yr) > 0:
        years   = na_yr.index.astype(int)
        ax_bar2 = ax_bar.twinx()
        bars    = ax_bar.bar(years, na_yr.values, color='#4C72B0',
                             edgecolor='white', linewidth=0.5, alpha=0.8, zorder=3)
        ax_bar2.plot(years, pct_yr.values, color='black', marker='o', markersize=3,
                     linewidth=1.2, linestyle='--', alpha=0.7)
        ax_bar2.set_ylabel('% missing', fontsize=8)
        ax_bar2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f %%'))
        ax_bar2.tick_params(labelsize=7)
        for bar, val in zip(bars, na_yr.values):
            if val > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(int(val)), ha='center', va='bottom', fontsize=6)
        ax_bar.set_ylabel('Missing values', fontsize=9)
        ax_bar.set_xticks(years)
        ax_bar.set_xticklabels(years, rotation=45, ha='right', fontsize=7)
        ax_bar.grid(axis='y', alpha=0.25)
        ax_bar.set_axisbelow(True)
    ax_bar.set_title('Missing Values per Year', fontsize=9,
                     fontweight='bold', loc='left', pad=4)

    pdf.savefig(fig, bbox_inches='tight')
    if figures_dir:
        fig.savefig(os.path.join(figures_dir, f'{var}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


# ── Public entry point ─────────────────────────────────────────────────────

def generate_quality_pdf(dataset_manager, station_info: pd.DataFrame,
                         output_dir: str = "out") -> str:
    """
    Run quality checks on all variables present in dataset_manager.data
    and write quality_checks.pdf to output_dir.
    Returns the path to the generated PDF, or empty string on failure.
    """
    df    = dataset_manager.data
    dates = df['DATE']

    # Global temporal checks
    dt_sorted  = dates.sort_values().diff().dropna()
    dt_mode    = dt_sorted.mode().iloc[0] if len(dt_sorted) else None
    n_gaps     = int((dt_sorted != dt_mode).sum()) if dt_mode is not None else 0
    n_dup      = int(dates.duplicated().sum())

    # Per-variable checks
    all_checks: dict = {}
    for var, cfg in VARIABLES.items():
        if var not in df.columns:
            continue
        chk = _check_variable(df[var], dates, cfg)
        all_checks[var] = chk
        n_flagged = int((chk['flags'] != '').sum())
        logger.info(
            "QC %-8s | missing=%d  absurd=%d  jumps=%d  plateau=%d  ctx=%d  "
            "flagged=%d (%.1f %%)",
            var, chk['n_missing'], chk['n_absurd'], chk['n_jumps'],
            chk['n_plateau'], chk['n_contextual'],
            n_flagged, chk['pct_missing'],
        )

    if not all_checks:
        logger.warning("No recognised variables found — quality PDF not generated.")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'quality_checks.pdf')

    with PdfPages(out_path) as pdf:
        _cover_page(pdf, df, station_info, all_checks, n_gaps, n_dup, figures_dir=figures_dir)
        for var, chk in all_checks.items():
            _variable_page(pdf, var, df[var], dates, VARIABLES[var], chk, figures_dir=figures_dir)

    logger.info("Quality report saved: %s", out_path)
    return out_path
