"""
=============================================================================
  Analyse de qualité et statistiques – Température à 2m
=============================================================================
  • Contrôle qualité : valeurs manquantes, doublons, valeurs absurdes,
    cohérence temporelle (sauts entre pas de temps).
  • Statistiques descriptives : moyenne, écart-type, quantiles,
    tendance de Mann-Kendall avec significativité.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings, textwrap
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, textwrap
from datetime import datetime

warnings.filterwarnings("ignore")


PATH_ROOT = 'C:\\Users\\SamyKraiem\\OneDrive - Callendar\\Bureau\\etude_orano'
os.chdir(PATH_ROOT)

# ── Configuration ──────────────────────────────────────────────────────────
FICHIER = "donnees_temperature.csv"
COL_DATE = "Date"
COL_TEMP = "Temperature"

# Seuils physiques raisonnables pour la température à 2 m (°C)
TEMP_MIN_ABS = -60.0   # record mondial ≈ -89 °C, on prend large
TEMP_MAX_ABS =  60.0   # record mondial ≈  56 °C, on prend large

# Variation max acceptable entre deux pas de temps consécutifs (°C)
DELTA_MAX = 6        # 10 °C en 10 min est déjà extrême
RAPPORT = "rapport_qualite.md"
SEUIL_CONSTANT = 144       # plateau CERTAIN : ≥ 24 h de valeur fixe (144 × 10 min)
SEUIL_PLATEAU_COURT = 24   # plateau SUSPECT : ≥ 4 h (24 × 10 min) + critère aggravant
 


# ── Classe pour construire le Markdown ────────────────────────────────────
class MarkdownReport:
    def __init__(self, path: str):
        self.path = path
        self.lines = []
 
    def h1(self, text):    self.lines.append(f"# {text}\n")
    def h2(self, text):    self.lines.append(f"## {text}\n")
    def h3(self, text):    self.lines.append(f"### {text}\n")
    def text(self, text):  self.lines.append(f"{text}\n")
    def blank(self):       self.lines.append("")
    def hr(self):          self.lines.append("---\n")
    def img(self, alt, path): self.lines.append(f"![{alt}]({path})\n")
 
    def table(self, headers, rows):
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(str(v) for v in row) + " |")
        self.lines.append("")
 
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"Rapport généré : {self.path}")
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(FICHIER, parse_dates=[COL_DATE])
df = df.sort_values(COL_DATE).reset_index(drop=True)
temp = df[COL_TEMP].copy()
 
dt = df[COL_DATE].diff()
dt_expected = dt.mode().iloc[0]
 
md = MarkdownReport(RAPPORT)
 
md.h1("Rapport de qualité et statistiques – Température à 2 m")
md.text(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*")
md.blank()
 
md.h2("1. Informations générales")
md.table(
    ["Paramètre", "Valeur"],
    [
        ["Fichier", f"`{FICHIER}`"],
        ["Lignes totales", f"{len(df):,}"],
        ["Période", f"{df[COL_DATE].min()} → {df[COL_DATE].max()}"],
        ["Pas de temps détecté", f"{dt_expected}"],
    ]
)
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  CONTRÔLE QUALITÉ
# ═══════════════════════════════════════════════════════════════════════════
md.h2("2. Contrôle qualité")
 
# ── Valeurs manquantes ────────────────────────────────────────────────────
md.h3("2.1 Valeurs manquantes")
n_missing = temp.isna().sum()
pct_missing = 100 * n_missing / len(temp)
md.text(f"**Nombre** : {n_missing}  ({pct_missing:.3f} %)")
 
if n_missing > 0:
    idx_missing = df.index[temp.isna()]
    dates_missing = df.loc[idx_missing, COL_DATE]
    md.text(f"- Première lacune : index {idx_missing[0]} ({dates_missing.iloc[0]})")
    md.text(f"- Dernière lacune : index {idx_missing[-1]} ({dates_missing.iloc[-1]})")
 
    is_na = temp.isna().astype(int)
    blocks = (is_na.diff().ne(0)).cumsum()
    na_blocks = is_na.groupby(blocks).agg(["sum", "first"])
    na_blocks = na_blocks[na_blocks["sum"] > 0]
    md.text(f"- Blocs consécutifs de NaN : **{len(na_blocks)}**")
    md.blank()
    top_blocks = na_blocks.nlargest(5, "sum")
    rows_blocks = []
    for _, row in top_blocks.iterrows():
        rows_blocks.append([f"~{row.name}", f"{int(row['sum'])}"])
    md.text("**5 plus grands blocs :**")
    md.table(["Index début (≈)", "Longueur (pas de temps)"], rows_blocks)
else:
    md.text("✅ Aucune valeur manquante détectée.")
md.blank()
 
# ── Graphiques valeurs manquantes ─────────────────────────────────────────
md.h3("2.2 Valeurs manquantes par année")
 
df["_year"] = df[COL_DATE].dt.year
df["_month"] = df[COL_DATE].dt.month
 
def saison(m):
    if m in [12, 1, 2]: return "Hiver (DJF)"
    if m in [3, 4, 5]:  return "Printemps (MAM)"
    if m in [6, 7, 8]:  return "Été (JJA)"
    return "Automne (SON)"
 
df["_saison"] = df["_month"].apply(saison)
df["_year_hiver"] = df["_year"].copy()
df.loc[df["_month"] == 12, "_year_hiver"] = df.loc[df["_month"] == 12, "_year"] + 1
 
def plot_missing_barplot(data, year_col, title, filename, color):
    na_par_an = data.groupby(year_col)[COL_TEMP].apply(lambda s: s.isna().sum())
    total_par_an = data.groupby(year_col)[COL_TEMP].size()
    pct_par_an = 100 * na_par_an / total_par_an
 
    fig, ax1 = plt.subplots(figsize=(12, 5))
    years = na_par_an.index.astype(int)
    bars = ax1.bar(years, na_par_an.values, color=color, edgecolor="white",
                   linewidth=0.5, alpha=0.85, zorder=3)
    ax2 = ax1.twinx()
    ax2.plot(years, pct_par_an.values, color="black", marker="o", markersize=4,
             linewidth=1.5, linestyle="--", alpha=0.7, zorder=4, label="% manquant")
    ax2.set_ylabel("% de valeurs manquantes", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for bar, val in zip(bars, na_par_an.values):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{int(val):,}".replace(",", " "),
                     ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax1.set_xlabel("Année", fontsize=11)
    ax1.set_ylabel("Nombre de valeurs manquantes", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(years)
    ax1.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
plot_missing_barplot(
    df, "_year", "Valeurs manquantes par année – Toutes saisons",
    "Figs/missing_annuel.png", color="#4C72B0")
md.text("**Toutes saisons :**")
md.img("Valeurs manquantes – Toutes saisons", "Figs/missing_annuel.png")

df_ete = df[df["_saison"] == "Été (JJA)"].copy()
plot_missing_barplot(
    df_ete, "_year", "Valeurs manquantes par année – Été (JJA)",
    "Figs/missing_ete.png", color="#DD8452")
md.text("**Été (JJA) :**")
md.img("Valeurs manquantes – Été", "Figs/missing_ete.png")

df_hiver = df[df["_saison"] == "Hiver (DJF)"].copy()
plot_missing_barplot(
    df_hiver, "_year_hiver", "Valeurs manquantes par année – Hiver (DJF)",
    "Figs/missing_hiver.png", color="#55A868")
md.text("**Hiver (DJF) :**")
md.img("Valeurs manquantes – Hiver", "Figs/missing_hiver.png")
 
# ── Doublons temporels ────────────────────────────────────────────────────
md.h3("2.3 Doublons temporels")
n_dup = df[COL_DATE].duplicated().sum()
md.text(f"**Horodatages dupliqués** : {n_dup}")
if n_dup > 0:
    dup_sample = df[df[COL_DATE].duplicated(keep=False)].head(10)
    rows_dup = [[str(r[COL_DATE]), f"{r[COL_TEMP]:.1f}"] for _, r in dup_sample.iterrows()]
    md.text("Exemples :")
    md.table(["Date", "Température (°C)"], rows_dup)
else:
    md.text("✅ Aucun doublon détecté.")
md.blank()
 
# ── Continuité temporelle ────────────────────────────────────────────────
md.h3("2.4 Continuité temporelle (trous)")
gaps = dt[dt != dt_expected].dropna()
md.text(f"**Pas de temps attendu** : {dt_expected}")
md.text(f"**Pas irréguliers** : {len(gaps)}")
if len(gaps) > 0:
    md.text("**10 plus grands trous :**")
    rows_gaps = []
    for i in gaps.nlargest(10).index:
        rows_gaps.append([
            str(df[COL_DATE].iloc[i-1]),
            str(df[COL_DATE].iloc[i]),
            str(dt.iloc[i])
        ])
    md.table(["De", "À", "Δt"], rows_gaps)
else:
    md.text("✅ Série temporelle continue.")
md.blank()
 
# ── Valeurs absurdes ─────────────────────────────────────────────────────
md.h3(f"2.5 Valeurs hors plage physique [{TEMP_MIN_ABS}, {TEMP_MAX_ABS}] °C")
mask_absurd = (temp < TEMP_MIN_ABS) | (temp > TEMP_MAX_ABS)
n_absurd = mask_absurd.sum()
md.text(f"**Nombre** : {n_absurd}")
if n_absurd > 0:
    rows_abs = [[str(r[COL_DATE]), f"{r[COL_TEMP]:.1f}"]
                for _, r in df.loc[mask_absurd, [COL_DATE, COL_TEMP]].head(10).iterrows()]
    md.table(["Date", "Température (°C)"], rows_abs)
else:
    md.text("✅ Toutes les valeurs sont dans la plage physique.")
md.blank()
 
# ── Cohérence temporelle ─────────────────────────────────────────────────
md.h3(f"2.6 Cohérence temporelle (|ΔT| > {DELTA_MAX} °C entre 2 pas)")
delta_temp = temp.diff().abs()
mask_jump = delta_temp > DELTA_MAX
n_jumps = mask_jump.sum()
md.text(f"**Sauts détectés** : {n_jumps}")
if n_jumps > 0:
    md.text("**10 plus gros sauts :**")
    rows_jump = []
    for i in delta_temp[mask_jump].nlargest(10).index:
        rows_jump.append([
            str(df[COL_DATE].iloc[i]),
            f"{delta_temp.iloc[i]:+.1f}",
            f"{temp.iloc[i-1]:.1f} → {temp.iloc[i]:.1f}"
        ])
    md.table(["Date", "ΔT (°C)", "Transition"], rows_jump)
else:
    md.text("✅ Aucun saut abrupt détecté.")
md.blank()
 
# ── Valeurs constantes prolongées ────────────────────────────────────────
_duree_h = SEUIL_CONSTANT * 10 // 60   # 144 × 10 min / 60 = 24 h
_duree_court_h = SEUIL_PLATEAU_COURT * 10 // 60
md.h3(f"2.7 Valeurs constantes prolongées (≥ {_duree_h} h)")
runs = (temp.diff().ne(0)).cumsum()
run_lengths = temp.groupby(runs).transform("count")
mask_flat = run_lengths >= SEUIL_CONSTANT
# Plateaux courts (4 h–24 h) — utilisés pour PLATEAU_SUSPECT dans le flagging
mask_flat_court = (run_lengths >= SEUIL_PLATEAU_COURT) & ~mask_flat
n_flat_pts = mask_flat.sum()

if n_flat_pts > 0:
    flat_groups = list(df.loc[mask_flat].groupby(runs[mask_flat]))
    md.text(f"**{len(flat_groups)} période(s)** de valeur constante ≥ {_duree_h} h :")
    rows_flat = []
    for name, grp in flat_groups[:10]:
        rows_flat.append([
            str(grp[COL_DATE].iloc[0]),
            str(grp[COL_DATE].iloc[-1]),
            f"{grp[COL_TEMP].iloc[0]:.1f}",
            str(len(grp))
        ])
    md.table(["Début", "Fin", "Valeur (°C)", "Nb pas"], rows_flat)
else:
    flat_groups = []
    md.text(f"✅ Aucune période constante ≥ {_duree_h} h détectée.")
md.blank()
 
# ── Résumé qualité ───────────────────────────────────────────────────────
md.h3("2.8 Résumé qualité")
flags = {
    "Valeurs manquantes": n_missing,
    "Doublons temporels": n_dup,
    "Pas irréguliers": len(gaps),
    "Valeurs absurdes": n_absurd,
    f"Sauts > {DELTA_MAX} °C": n_jumps,
    f"Périodes constantes ≥ {_duree_h} h": n_flat_pts,
}
rows_flags = []
for k, v in flags.items():
    status = "✅" if v == 0 else "⚠️"
    rows_flags.append([status, k, str(v)])
md.table(["Statut", "Test", "Nombre"], rows_flags)


# ═══════════════════════════════════════════════════════════════════════════
#  2bis.  FLAGGING DES DONNÉES SUSPECTES
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  FLAGGING")
print("=" * 72)

FICHIER_FLAGGED = "donnees_temperature_flagged.csv"

# ── Construction du masque JUMP étendu ────────────────────────────────────
# On flagge les DEUX extrémités de chaque saut (i et i-1) car on ne peut
# pas déterminer a priori laquelle des deux valeurs est erronée.
jump_idx_set = set(df.index[mask_jump].tolist())
jump_prev_idx = {i - 1 for i in jump_idx_set if i - 1 >= df.index[0]}
mask_jump_ext = pd.Series(False, index=df.index)
mask_jump_ext[list(jump_idx_set | jump_prev_idx)] = True

# ── PLATEAU_SUSPECT : plateau ≥ 4h avec critère aggravant ────────────────
# Critères : JUMP à l'entrée ou à la sortie du plateau OU valeur == 0.0 °C
# Couvre à la fois les plateaux longs (PLATEAU) et courts (4h–24h).
def _plateau_suspect(grp):
    """True si le plateau est contextuallement suspect."""
    start_i = grp.index[0]
    end_i   = grp.index[-1]
    jump_in  = start_i in jump_idx_set
    iloc_end = df.index.get_loc(end_i)
    jump_out = (iloc_end + 1 < len(df)) and (df.index[iloc_end + 1] in jump_idx_set)
    val      = grp[COL_TEMP].iloc[0]
    is_zero  = pd.notna(val) and abs(val) < 0.05
    return jump_in or jump_out or is_zero

plateau_suspect_pts = set()
# Plateaux certains (≥ 24h)
for _, grp in flat_groups:
    if _plateau_suspect(grp):
        plateau_suspect_pts.update(grp.index.tolist())
# Plateaux courts (4h–24h)
flat_court_groups = (list(df.loc[mask_flat_court].groupby(runs[mask_flat_court]))
                     if mask_flat_court.sum() > 0 else [])
for _, grp in flat_court_groups:
    if _plateau_suspect(grp):
        plateau_suspect_pts.update(grp.index.tolist())

mask_plateau_suspect = pd.Series(False, index=df.index)
if plateau_suspect_pts:
    mask_plateau_suspect.loc[list(plateau_suspect_pts)] = True

# ── CONTEXTUAL_SUSPECT : valeur hors bornes climatologiques ──────────────
# Pour chaque combinaison (mois, heure), on calcule Q1−3×IQR et Q3+3×IQR
# sur l'ensemble de la série. Une valeur hors de ces bornes est signalée.
df["_hour"] = df[COL_DATE].dt.hour
q_grp = df.groupby(["_month", "_hour"])[COL_TEMP].quantile([0.25, 0.75]).unstack()
q_grp.columns = ["q25", "q75"]
q_grp["iqr"]   = q_grp["q75"] - q_grp["q25"]
q_grp["lower"] = q_grp["q25"] - 3 * q_grp["iqr"]
q_grp["upper"] = q_grp["q75"] + 3 * q_grp["iqr"]

mh_idx = pd.MultiIndex.from_arrays([df["_month"], df["_hour"]])
ctx_lower = q_grp["lower"].reindex(mh_idx).values
ctx_upper = q_grp["upper"].reindex(mh_idx).values

mask_contextual = pd.Series(
    (temp.values < ctx_lower) | (temp.values > ctx_upper),
    index=df.index
) & temp.notna() & ~mask_absurd   # ABSURD est déjà couvert par son propre flag

# ── Attribution des flags ─────────────────────────────────────────────────
flag_col = pd.Series("", index=df.index, dtype=object)

for mask, code in [
    (temp.isna(),          "MISSING"),
    (mask_absurd,          "ABSURD"),
    (mask_jump_ext,        "JUMP"),
    (mask_flat,            "PLATEAU"),
    (mask_plateau_suspect, "PLATEAU_SUSPECT"),
    (mask_contextual,      "CONTEXTUAL_SUSPECT"),
]:
    m_empty  = mask & (flag_col == "")
    m_filled = mask & (flag_col != "")
    flag_col.loc[m_empty]  = code
    flag_col.loc[m_filled] = flag_col.loc[m_filled] + "|" + code

df["flag"] = flag_col

# ── Export ────────────────────────────────────────────────────────────────
df[[COL_DATE, COL_TEMP, "flag"]].to_csv(FICHIER_FLAGGED, index=False)

# ── Résumé console ────────────────────────────────────────────────────────
n_flagged   = (flag_col != "").sum()
n_clean     = (flag_col == "").sum()
pct_flagged = 100 * n_flagged / len(df)

code_counts = (flag_col[flag_col != ""]
               .str.split("|").explode()
               .value_counts().sort_index())

print(f"  Pas de temps flagges  : {n_flagged:,} ({pct_flagged:.3f} %)")
print(f"  Pas de temps propres  : {n_clean:,}")
print(f"  Detail par code :")
for code, cnt in code_counts.items():
    print(f"    {code:<20} : {cnt:,}")
print(f"  Fichier exporte : {FICHIER_FLAGGED}")

# ── Rapport Markdown ──────────────────────────────────────────────────────
md.h3("2.9 Flagging des données suspectes")
md.text("Codes attribués à chaque pas de temps défaillant :")
md.table(
    ["Code", "Critère", "Niveau"],
    [
        ["`MISSING`",            "Valeur manquante (NaN)",
         "Certain"],
        ["`ABSURD`",             f"Hors plage physique [{TEMP_MIN_ABS}, {TEMP_MAX_ABS}] °C",
         "Certain"],
        ["`JUMP`",               f"|ΔT| > {DELTA_MAX} °C entre deux pas consécutifs — les deux extrémités flaggées",
         "Certain"],
        ["`PLATEAU`",            f"Valeur fixe ≥ {_duree_h} h consécutives",
         "Certain"],
        ["`PLATEAU_SUSPECT`",    f"Valeur fixe ≥ {_duree_court_h} h ET (JUMP adjacent OU valeur == 0,0 °C)",
         "Suspect"],
        ["`CONTEXTUAL_SUSPECT`", "Valeur hors Q1−3×IQR / Q3+3×IQR pour son (mois, heure)",
         "Suspect"],
    ]
)
md.text(f"**Total flaggé** : {n_flagged:,} pas de temps ({pct_flagged:.3f} %)")
md.text(f"**Propres** : {n_clean:,} pas de temps")
md.blank()
md.table(
    ["Code", "Nombre de pas de temps"],
    [[code, f"{cnt:,}"] for code, cnt in code_counts.items()]
)
md.text(f"Fichier exporté : `{FICHIER_FLAGGED}`")
md.blank()


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTIQUES DESCRIPTIVES
# ═══════════════════════════════════════════════════════════════════════════
md.h2("3. Statistiques descriptives")
 
temp_clean = temp.dropna()
md.table(
    ["Statistique", "Valeur"],
    [
        ["N (valide)", f"{len(temp_clean):,}"],
        ["Moyenne", f"{temp_clean.mean():.4f} °C"],
        ["Écart-type", f"{temp_clean.std():.4f} °C"],
        ["Min", f"{temp_clean.min():.2f} °C"],
        ["Max", f"{temp_clean.max():.2f} °C"],
        ["Médiane", f"{temp_clean.median():.2f} °C"],
        ["Quantile 0.01", f"{temp_clean.quantile(0.01):.2f} °C"],
        ["Quantile 0.05", f"{temp_clean.quantile(0.05):.2f} °C"],
        ["Quantile 0.10", f"{temp_clean.quantile(0.10):.2f} °C"],
        ["Quantile 0.90", f"{temp_clean.quantile(0.90):.2f} °C"],
        ["Quantile 0.95", f"{temp_clean.quantile(0.95):.2f} °C"],
        ["Quantile 0.99", f"{temp_clean.quantile(0.99):.2f} °C"],
        ["Asymétrie (skew)", f"{temp_clean.skew():.4f}"],
        ["Kurtosis", f"{temp_clean.kurtosis():.4f}"],
    ]
)
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  TENDANCE – MANN-KENDALL
# ═══════════════════════════════════════════════════════════════════════════
md.h2("4. Test de Mann-Kendall (tendance)")
 
def mann_kendall(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    s = 0
    for k in range(n - 1):
        diff = x[k + 1:] - x[k]
        s += np.sum(np.sign(diff))
    unique, counts = np.unique(x, return_counts=True)
    tp = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    if p < alpha:
        trend = "croissante ↑" if s > 0 else "décroissante ↓"
    else:
        trend = "pas de tendance"
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    slope = np.median(slopes) if slopes else float("nan")
    return s, z, p, trend, slope
 
md.text("*Test appliqué sur les moyennes journalières.*")
md.blank()
 
df["_date_only"] = df[COL_DATE].dt.date
daily = df.groupby("_date_only")[COL_TEMP].mean().dropna()
 
MAX_PTS = 20_000
if len(daily) > MAX_PTS:
    idx = np.linspace(0, len(daily) - 1, MAX_PTS, dtype=int)
    series_mk = daily.values[idx]
    md.text(f"> Sous-échantillonnage à {MAX_PTS:,} points (série originale : {len(daily):,} jours).")
else:
    series_mk = daily.values
 
S, Z, p, trend, slope = mann_kendall(series_mk)
 
if p < 0.001:
    sig = "\\*\\*\\* (p < 0.001) – très hautement significatif"
elif p < 0.01:
    sig = "\\*\\* (p < 0.01) – hautement significatif"
elif p < 0.05:
    sig = "\\* (p < 0.05) – significatif"
else:
    sig = "ns (p ≥ 0.05) – non significatif"
 
md.table(
    ["Paramètre", "Valeur"],
    [
        ["Jours disponibles", f"{len(daily):,}"],
        ["S (statistique)", f"{S:,.0f}"],
        ["Z (normalisé)", f"{Z:.4f}"],
        ["p-value", f"{p:.2e}"],
        ["Tendance", trend],
        ["Pente de Sen", f"{slope:.6f} °C/jour"],
        ["Pente de Sen (≈/an)", f"{slope * 365.25:.4f} °C/an"],
        ["Significativité", sig],
    ]
)
 
 
# ── Figure tendance ───────────────────────────────────────────────────────
fig_trend = "Figs/tendance.png"

# Moyennes et écarts-types annuels
annual = df.groupby("_year")[COL_TEMP].agg(mean="mean", std="std").dropna()
years_ann = annual.index.astype(int)

# Droite de Sen calée sur les moyennes annuelles
sen_line = annual["mean"].iloc[0] + slope * 365.25 * (years_ann - years_ann[0])

fig, ax = plt.subplots(figsize=(14, 5))

# Enveloppe ±1σ annuelle
ax.fill_between(years_ann,
                annual["mean"] - annual["std"],
                annual["mean"] + annual["std"],
                alpha=0.18, color="#4C72B0", label="±1 σ annuel")

# Moyennes annuelles
ax.plot(years_ann, annual["mean"], color="#4C72B0", linewidth=1.8,
        marker="o", markersize=4, label="Moyenne annuelle")

# Moyenne mobile 5 ans
if len(annual) >= 5:
    rolling = annual["mean"].rolling(5, center=True).mean()
    ax.plot(years_ann, rolling, color="#DD8452", linewidth=2.2,
            linestyle="--", label="Moyenne mobile 5 ans")

# Droite de tendance (pente de Sen)
ax.plot(years_ann, sen_line, color="crimson", linewidth=2,
        linestyle="-", label=f"Tendance Sen : {slope * 365.25:+.3f} °C/an")

# Annotation significativité
sig_label = "p < 0,001 ***" if p < 0.001 else (
            "p < 0,01 **" if p < 0.01 else (
            "p < 0,05 *" if p < 0.05 else "ns"))
ax.text(0.02, 0.96,
        f"Mann-Kendall  |  {sig_label}  |  pente Sen ≈ {slope * 365.25:+.3f} °C/an",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

ax.set_xlabel("Année", fontsize=11)
ax.set_ylabel("Température (°C)", fontsize=11)
ax.set_title("Tendance de la température à 2 m – Tricastin (1990–2026)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(years_ann[0] - 0.5, years_ann[-1] + 0.5)
fig.tight_layout()
fig.savefig(fig_trend, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    OK {fig_trend}")

md.blank()
md.text("**Figure – Tendance temporelle :**")
md.img("Tendance température", fig_trend)


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTIQUES SAISONNIÈRES
# ═══════════════════════════════════════════════════════════════════════════
md.h2("5. Statistiques saisonnières")
 
for s in ["Hiver (DJF)", "Printemps (MAM)", "Été (JJA)", "Automne (SON)"]:
    sub = df.loc[df["_saison"] == s, COL_TEMP].dropna()
    md.h3(s)
    if len(sub) == 0:
        md.text("*Aucune donnée pour cette saison.*")
        continue
    md.table(
        ["Statistique", "Valeur"],
        [
            ["N", f"{len(sub):,}"],
            ["Moyenne", f"{sub.mean():.2f} °C"],
            ["Écart-type", f"{sub.std():.2f} °C"],
            ["Min", f"{sub.min():.1f} °C"],
            ["Max", f"{sub.max():.1f} °C"],
            ["Q05", f"{sub.quantile(0.05):.2f} °C"],
            ["Q95", f"{sub.quantile(0.95):.2f} °C"],
        ]
    )
 
# ── Nettoyage colonnes internes ───────────────────────────────────────────
df.drop(columns=["_date_only", "_month", "_hour", "_saison", "_year", "_year_hiver"],
        inplace=True, errors="ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  6. AGRÉGATION JOURNALIÈRE — TASMAX / TASMIN / TAS
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  AGREGATION JOURNALIERE")
print("=" * 72)

DATA_4_BC = os.path.join(PATH_ROOT, "data_4_BC")
os.makedirs(DATA_4_BC, exist_ok=True)

# ── Stratégie de rejet ────────────────────────────────────────────────────
FLAGS_REJETES = {"MISSING", "ABSURD", "PLATEAU", "PLATEAU_SUSPECT", "JUMP"}
# CONTEXTUAL_SUSPECT conservé : physiquement plausible, potentiellement
# un extrême réel. Signalé dans les métadonnées.
MIN_PAS_PAR_HEURE  = 3    # ≥ 30 min valides sur 60 min  (6 pas × 10 min)
MIN_HEURES_PAR_JOUR = 18  # ≥ 75 % des heures valides sur 24 h

# ── Masque des pas à exclure ──────────────────────────────────────────────
# Vectorisé : explode les codes, repère ceux à rejeter (correspondance exacte)
rejected_idx = (
    df["flag"].str.split("|").explode()
    .pipe(lambda s: s[s.isin(FLAGS_REJETES)])
    .index.unique()
)
mask_rej = pd.Series(False, index=df.index)
if len(rejected_idx):
    mask_rej.loc[rejected_idx] = True

temp_valid = df[COL_TEMP].where(~mask_rej)

# ── Moyennes horaires ─────────────────────────────────────────────────────
h_floor = df[COL_DATE].dt.floor("h")
h_count = (~mask_rej).groupby(h_floor).sum()   # nb pas valides par heure
h_mean  = temp_valid.groupby(h_floor).mean()   # moyenne horaire brute
h_mean  = h_mean.where(h_count >= MIN_PAS_PAR_HEURE)

# ── Agrégation journalière ─────────────────────────────────────────────────
d_idx    = h_mean.index.normalize()
n_h_ok   = h_mean.notna().groupby(d_idx).sum()
enough   = n_h_ok >= MIN_HEURES_PAR_JOUR
tasmax_s = h_mean.groupby(d_idx).max().where(enough)
tasmin_s = h_mean.groupby(d_idx).min().where(enough)
tas_s    = h_mean.groupby(d_idx).mean().where(enough)

# ── Troncature à fin 2025 ─────────────────────────────────────────────────
DATE_FIN_EXPORT = pd.Timestamp("2025-12-31")
tasmax_s = tasmax_s[tasmax_s.index <= DATE_FIN_EXPORT]
tasmin_s = tasmin_s[tasmin_s.index <= DATE_FIN_EXPORT]
tas_s    = tas_s[tas_s.index    <= DATE_FIN_EXPORT]

# ── Grille calendaire complète ─────────────────────────────────────────────
full_idx = pd.date_range(tasmax_s.index.min(), tasmax_s.index.max(), freq="D")
tasmax_s = tasmax_s.reindex(full_idx)
tasmin_s = tasmin_s.reindex(full_idx)
tas_s    = tas_s.reindex(full_idx)

# ── Export CSV ─────────────────────────────────────────────────────────────
for var, series in [("tasmax", tasmax_s), ("tasmin", tasmin_s), ("tas", tas_s)]:
    out_path = os.path.join(DATA_4_BC, f"{var}_historical.csv")
    pd.DataFrame({
        "time": full_idx.strftime("%Y-%m-%d 12:00:00"),
        "obs":  series.round(2).values,
    }).to_csv(out_path, index=False)
    n_miss = series.isna().sum()
    pct_miss = 100 * n_miss / len(series)
    print(f"  {var}_historical.csv : {len(series):,} jours, {n_miss} manquants ({pct_miss:.2f} %)")


# ── Section 6 du rapport Markdown ─────────────────────────────────────────
def _missing_ranges(series, idx):
    """Regroupe les dates manquantes en plages consécutives."""
    miss = idx[series.isna().values]
    if len(miss) == 0:
        return []
    ranges = []
    s = p = miss[0]
    for d in miss[1:]:
        if (d - p).days == 1:
            p = d
        else:
            ranges.append((s, p))
            s = p = d
    ranges.append((s, p))
    return ranges

md.h2("6. Agrégation journalière – tasmax / tasmin / tas")
md.text("**Stratégie de rejet des flags :**")
md.table(
    ["Flag", "Rejeté ?", "Justification"],
    [
        ["`MISSING`",            "Oui", "Valeur manquante"],
        ["`ABSURD`",             "Oui", "Hors plage physique"],
        ["`PLATEAU`",            "Oui", "Capteur bloqué ≥ 24 h"],
        ["`PLATEAU_SUSPECT`",    "Oui", "Capteur probablement bloqué ≥ 4 h"],
        ["`JUMP`",               "Oui", "Transition aberrante — biais potentiel sur max/min"],
        ["`CONTEXTUAL_SUSPECT`", "**Non**", "Physiquement plausible ; peut être un extrême réel"],
    ]
)
md.text(
    f"**Règles d'agrégation :** "
    f"≥ {MIN_PAS_PAR_HEURE}/6 pas valides pour la moyenne horaire · "
    f"≥ {MIN_HEURES_PAR_JOUR}/24 heures valides pour la valeur journalière."
)
md.blank()

var_meta = {
    "tasmax": ("Température maximale journalière",
               "Max des moyennes horaires (10 min → 1 h → max/24 h)"),
    "tasmin": ("Température minimale journalière",
               "Min des moyennes horaires (10 min → 1 h → min/24 h)"),
    "tas":    ("Température moyenne journalière",
               "Moyenne des moyennes horaires (10 min → 1 h → moy/24 h)"),
}

for i, (var, series) in enumerate(
        [("tasmax", tasmax_s), ("tasmin", tasmin_s), ("tas", tas_s)], start=1):
    label, aggr = var_meta[var]
    n_tot  = len(series)
    n_miss = series.isna().sum()
    ranges = _missing_ranges(series, full_idx)

    md.h3(f"6.{i} `{var}_historical.csv` — {label}")
    md.table(
        ["Paramètre", "Valeur"],
        [
            ["Fichier",         f"`data_4_BC/{var}_historical.csv`"],
            ["Période",         f"{full_idx[0].date()} → {full_idx[-1].date()}"],
            ["Agrégation",      aggr],
            ["Jours total",     f"{n_tot:,}"],
            ["Jours valides",   f"{n_tot - n_miss:,}"],
            ["Jours manquants", f"{n_miss} ({100*n_miss/n_tot:.2f} %)"],
        ]
    )
    if n_miss > 0:
        md.text(f"**Plages manquantes ({len(ranges)} période(s)) :**")
        rows_miss = []
        for rs, re in ranges[:30]:
            label_r = str(rs.date()) if rs == re else f"{rs.date()} → {re.date()}"
            rows_miss.append([label_r, str((re - rs).days + 1)])
        md.table(["Période", "Nb jours"], rows_miss)
        if len(ranges) > 30:
            md.text(f"> *... {len(ranges) - 30} plages supplémentaires non affichées.*")
    else:
        md.text("Aucun jour manquant.")
    md.blank()


# ── Fiches descriptives individuelles (.md) ───────────────────────────────
for var, series in [("tasmax", tasmax_s), ("tasmin", tasmin_s), ("tas", tas_s)]:
    label, aggr = var_meta[var]
    n_tot  = len(series)
    n_miss = series.isna().sum()
    ranges = _missing_ranges(series, full_idx)

    lines = [
        f"# {var}_historical — Fiche descriptive",
        "",
        "## Identification",
        "",
        "| Champ | Valeur |",
        "| --- | --- |",
        f"| Variable | `{var}` |",
        f"| Nom complet (CF) | {label} |",
        f"| Unité | °C |",
        f"| Fichier | `data_4_BC/{var}_historical.csv` |",
        f"| Site | Tricastin (44.34236°N, 4.7266°E) |",
        f"| Hauteur capteur | Sol (0 m) |",
        f"| Source brute | `donnees_temperature_flagged.csv` |",
        f"| Pas de temps source | 10 min |",
        "",
        "## Période",
        "",
        "| Début | Fin | Jours total | Jours valides | Jours manquants |",
        "| --- | --- | --- | --- | --- |",
        (f"| {full_idx[0].date()} | {full_idx[-1].date()} "
         f"| {n_tot:,} | {n_tot - n_miss:,} "
         f"| {n_miss} ({100*n_miss/n_tot:.2f} %) |"),
        "",
        "## Agrégation",
        "",
        aggr,
        "",
        f"- Seuil horaire : ≥ {MIN_PAS_PAR_HEURE}/6 pas de 10 min valides (≥ 30 min)",
        f"- Seuil journalier : ≥ {MIN_HEURES_PAR_JOUR}/24 heures valides (≥ 75 %)",
        "",
        "## Flags rejetés lors de l'agrégation",
        "",
        "| Flag | Motif du rejet |",
        "| --- | --- |",
        "| `MISSING` | Valeur manquante (NaN) |",
        "| `ABSURD` | Hors plage physique [−60, 60] °C |",
        "| `PLATEAU` | Capteur bloqué ≥ 24 h |",
        "| `PLATEAU_SUSPECT` | Plateau ≥ 4 h avec JUMP adjacent ou valeur == 0,0 °C |",
        "| `JUMP` | \\|ΔT\\| > 6 °C entre deux pas — biais potentiel sur l'extremum |",
        "",
        ("> **`CONTEXTUAL_SUSPECT` conservé** : valeur hors Q1−3×IQR / Q3+3×IQR "
         "pour son (mois, heure),  "),
        "> mais physiquement plausible. Peut correspondre à un extrême réel.",
        "",
    ]

    if n_miss == 0:
        lines += ["## Jours manquants", "", "Aucun jour manquant.", ""]
    else:
        lines += [
            f"## Jours manquants ({n_miss} jours — {len(ranges)} plage(s))",
            "",
            "| Période | Nb jours |",
            "| --- | --- |",
        ]
        for rs, re in ranges:
            label_r = str(rs.date()) if rs == re else f"{rs.date()} → {re.date()}"
            lines.append(f"| {label_r} | {(re - rs).days + 1} |")
        lines.append("")

    lines += [
        "---",
        f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} par `data_check_v1.py`*",
        "",
    ]

    note_path = os.path.join(DATA_4_BC, f"{var}_historical.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Note    : {var}_historical.md")


# ═══════════════════════════════════════════════════════════════════════════
#  6b. CONTRÔLES COMPLÉMENTAIRES
# ═══════════════════════════════════════════════════════════════════════════

# ── Test de Pettitt (homogénéité) sur les moyennes annuelles ──────────────
# Détecte un unique point de rupture dans la série (changement de capteur,
# déplacement de station). Basé sur la statistique U_t = Σ_i<t Σ_j>t sgn(x_j - x_i).
md.h2("6b. Contrôles complémentaires")
md.h3("6b.1 Test de Pettitt – homogénéité de la série")
md.text("*Test non-paramétrique de détection d'une rupture unique dans les moyennes annuelles journalières.*")

df_daily_full = pd.DataFrame({
    "date":  full_idx,
    "tasmax": tasmax_s.values,
    "tasmin": tasmin_s.values,
    "tas":    tas_s.values,
})
df_daily_full["year"] = df_daily_full["date"].dt.year
annual_mean = df_daily_full.groupby("year")["tas"].mean().dropna()
# Exclure la dernière année si incomplète (< 330 jours valides) pour ne pas
# biaiser la rupture vers la fin de série
year_counts = df_daily_full.groupby("year")["tas"].count()
last_year = year_counts.index[-1]
if year_counts[last_year] < 330:
    annual_mean = annual_mean.drop(last_year, errors="ignore")
x_pettitt = annual_mean.values
n_p = len(x_pettitt)

# Calcul de U_t pour t = 1..n-1
U = np.zeros(n_p)
for t in range(1, n_p):
    U[t] = U[t-1] + np.sum(np.sign(x_pettitt[t] - x_pettitt[:t])) \
                  - np.sum(np.sign(x_pettitt[:t] - x_pettitt[t]))
K = int(np.argmax(np.abs(U)))         # indice de la rupture candidate
K_stat = np.abs(U[K])                 # statistique K
# p-value approchée (Pettitt 1979)
p_pettitt = 2 * np.exp(-6 * K_stat**2 / (n_p**3 + n_p**2))
p_pettitt = min(p_pettitt, 1.0)
year_break = int(annual_mean.index[K])
sig_p = "significatif (p < 0,05) — rupture probable" if p_pettitt < 0.05 else "non significatif (p ≥ 0,05)"

md.table(
    ["Paramètre", "Valeur"],
    [
        ["N années", str(n_p)],
        ["Statistique K", f"{K_stat:.1f}"],
        ["p-value (approx.)", f"{p_pettitt:.4f}"],
        ["Significativité", sig_p],
        ["Année de rupture candidate", str(year_break)],
    ]
)
last_year_pettitt = int(annual_mean.index[-1])
end_artifact = year_break >= last_year_pettitt - 4   # rupture dans les 5 dernières années

if p_pettitt < 0.05 and not end_artifact:
    md.text(
        f"> ⚠️ Rupture détectée vers **{year_break}**. "
        "Vérifier un éventuel changement de capteur ou de station à cette date. "
        "Une série inhomogène peut biaiser la correction de biais."
    )
elif p_pettitt < 0.05 and end_artifact:
    md.text(
        f"> ℹ️ Rupture candidate en **{year_break}** (fin de série) — "
        "**artefact de tendance** : avec une tendance monotone confirmée (Mann-Kendall), "
        "la statistique U_t de Pettitt croît jusqu'au dernier point et détecte "
        "systématiquement la fin de la série. Pas de rupture instrumentale identifiée."
    )
else:
    md.text("Aucune rupture significative détectée — série homogène.")
md.blank()

print(f"  Pettitt : K={K_stat:.1f}, p={p_pettitt:.4f}, rupture candidate={year_break}")

# ── Cohérence interne tasmax ≥ tas ≥ tasmin ──────────────────────────────
md.h3("6b.2 Cohérence interne tasmax ≥ tas ≥ tasmin")

valid_days = df_daily_full.dropna(subset=["tasmax", "tas", "tasmin"])
viol_max_tas = (valid_days["tasmax"] < valid_days["tas"]).sum()
viol_tas_min = (valid_days["tas"] < valid_days["tasmin"]).sum()
viol_max_min = (valid_days["tasmax"] < valid_days["tasmin"]).sum()
n_valid = len(valid_days)

md.table(
    ["Violation", "Nb jours", "% jours valides"],
    [
        ["tasmax < tas",    str(viol_max_tas), f"{100*viol_max_tas/n_valid:.3f} %"],
        ["tas < tasmin",    str(viol_tas_min), f"{100*viol_tas_min/n_valid:.3f} %"],
        ["tasmax < tasmin", str(viol_max_min), f"{100*viol_max_min/n_valid:.3f} %"],
    ]
)
n_viol_total = viol_max_tas + viol_tas_min + viol_max_min
if n_viol_total == 0:
    md.text("Aucune violation détectée — cohérence interne validée.")
else:
    md.text(
        f"> ⚠️ {n_viol_total} violation(s) détectée(s). "
        "Ces jours peuvent introduire des incohérences dans la correction de biais."
    )
md.blank()

print(f"  Coherence tasmax>=tas>=tasmin : {n_viol_total} violation(s)")


# ═══════════════════════════════════════════════════════════════════════════
#  7. FIGURES DE SYNTHÈSE — RÉPARTITION DES FLAGS
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  FIGURES SYNTHESE FLAGS")
print("=" * 72)

FLAG_COLORS = {
    "Clean":              "#2ecc71",
    "MISSING":            "#95a5a6",
    "ABSURD":             "#c0392b",
    "PLATEAU":            "#e67e22",
    "PLATEAU_SUSPECT":    "#f1c40f",
    "JUMP":               "#8e44ad",
    "CONTEXTUAL_SUSPECT": "#2980b9",
}

# ── Figure 1 : données 10 min ─────────────────────────────────────────────
# Camembert gauche : Clean vs Total flaggé
# Camembert droit  : détail des données flaggées seulement

FIG_10MIN = "Figs/flags_10min.png"

n_total = len(df)
flag_order = ["MISSING", "PLATEAU", "PLATEAU_SUSPECT", "JUMP",
              "CONTEXTUAL_SUSPECT", "ABSURD"]
flag_vals  = [int(code_counts.get(c, 0)) for c in flag_order]
flag_cols  = [FLAG_COLORS[c] for c in flag_order]
# Filtrer les codes à zéro
nz = [(c, v, col) for c, v, col in zip(flag_order, flag_vals, flag_cols) if v > 0]
flag_order_nz, flag_vals_nz, flag_cols_nz = zip(*nz) if nz else ([], [], [])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Contrôle qualité — Série 10 min  |  Tricastin 1990–2026\n"
    f"N = {n_total:,} pas de temps",
    fontsize=13, fontweight="bold", y=1.02
)

# ── Gauche : vue globale ───────────────────────────────────────────────────
global_sizes  = [int(n_clean), int(n_flagged)]
global_colors = [FLAG_COLORS["Clean"], "#e74c3c"]
global_labels = [
    f"Propres\n{n_clean:,}  ({100*n_clean/n_total:.2f} %)",
    f"Flaggés\n{n_flagged:,}  ({100*n_flagged/n_total:.2f} %)",
]
wedges1, texts1 = ax1.pie(
    global_sizes,
    colors=global_colors,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    explode=[0, 0.06],
)
ax1.set_title("Vue globale\n(propres vs flaggés)", fontsize=11, fontweight="bold")
ax1.legend(wedges1, global_labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False)

# ── Droite : détail des flaggés ────────────────────────────────────────────
total_flags_counted = sum(flag_vals_nz)
legend_detail = [
    f"{c}  ({v:,} — {100*v/int(n_flagged):.1f} %)"
    for c, v in zip(flag_order_nz, flag_vals_nz)
]
wedges2, _, autotexts2 = ax2.pie(
    flag_vals_nz,
    colors=flag_cols_nz,
    startangle=90,
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    wedgeprops=dict(edgecolor="white", linewidth=2),
    pctdistance=0.72,
)
for at in autotexts2:
    at.set_fontsize(8.5)
ax2.set_title(
    f"Détail des données flaggées\n"
    f"({n_flagged:,} pas — multi-flag : comptage par code)",
    fontsize=11, fontweight="bold"
)
ax2.legend(wedges2, legend_detail, loc="lower center",
           bbox_to_anchor=(0.5, -0.26), fontsize=9, frameon=False,
           ncol=1)

fig.tight_layout()
fig.savefig(FIG_10MIN, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure : {FIG_10MIN}")

# ── Figure 2 : agrégation journalière ─────────────────────────────────────
FIG_DAILY = "Figs/flags_journalier.png"

daily_vars_list = [
    ("tasmax", tasmax_s, "tasmax\n(max horaire)"),
    ("tasmin", tasmin_s, "tasmin\n(min horaire)"),
    ("tas",    tas_s,    "tas\n(moy horaire)"),
]

fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(
    "Couverture journalière après agrégation  |  Tricastin 1990–2026\n"
    f"N = {len(full_idx):,} jours  ·  Seuil : ≥ {MIN_HEURES_PAR_JOUR}/24 h valides",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, (var, series, title) in zip(axes, daily_vars_list):
    n_tot  = len(series)
    n_miss = series.isna().sum()
    n_ok   = n_tot - n_miss

    slices = [n_ok, n_miss]
    clrs   = [FLAG_COLORS["Clean"], "#e74c3c"]
    lbls   = [
        f"Valides\n{n_ok:,}  ({100*n_ok/n_tot:.1f} %)",
        f"Manquants\n{n_miss}  ({100*n_miss/n_tot:.1f} %)",
    ]
    wedges_d, _, autotexts_d = ax.pie(
        slices,
        colors=clrs,
        startangle=90,
        autopct="%1.1f%%",
        wedgeprops=dict(edgecolor="white", linewidth=2),
        pctdistance=0.72,
        explode=[0, 0.06],
    )
    for at in autotexts_d:
        at.set_fontsize(10, )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(wedges_d, lbls, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=9, frameon=False)

fig2.tight_layout()
fig2.savefig(FIG_DAILY, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Figure : {FIG_DAILY}")

# ── Section 7 du rapport ──────────────────────────────────────────────────
md.h2("7. Synthèse graphique des flags")
md.h3("7.1 Série 10 min")
md.img("Répartition des flags – 10 min", FIG_10MIN)
md.blank()
md.h3("7.2 Agrégation journalière")
md.img("Couverture journalière", FIG_DAILY)
md.blank()


# ═══════════════════════════════════════════════════════════════════════════
#  8. CDF DES SÉRIES JOURNALIÈRES
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  CDF JOURNALIERE")
print("=" * 72)

FIG_CDF = "Figs/cdf_journalier.png"

cdf_vars = [
    ("tasmax", tasmax_s, "#c0392b", "tasmax — Maximum journalier"),
    ("tas",    tas_s,    "#27ae60", "tas — Moyenne journalière"),
    ("tasmin", tasmin_s, "#2980b9", "tasmin — Minimum journalier"),
]

fig_cdf, ax_cdf = plt.subplots(figsize=(11, 6))

for var, series, color, label in cdf_vars:
    vals = np.sort(series.dropna().values)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax_cdf.plot(vals, cdf * 100, color=color, linewidth=2, label=f"{label}  (N={len(vals):,})")

    # Repères Q05 / Q50 / Q95
    for q, ls in [(0.05, ":"), (0.50, "--"), (0.95, ":")]:
        xq = np.quantile(vals, q)
        yq = q * 100
        ax_cdf.plot(xq, yq, marker="o", color=color, markersize=5, zorder=5)
        if var == "tas":   # annoter une seule fois pour ne pas surcharger
            ax_cdf.axvline(xq, color="grey", linewidth=0.6, linestyle=ls, alpha=0.5)
            ax_cdf.axhline(yq, color="grey", linewidth=0.6, linestyle=ls, alpha=0.5)
            ax_cdf.text(xq + 0.3, yq + 1.2, f"Q{int(q*100):02d}", fontsize=7.5,
                        color="grey", va="bottom")

ax_cdf.set_xlabel("Température (°C)", fontsize=12)
ax_cdf.set_ylabel("Fréquence cumulée (%)", fontsize=12)
ax_cdf.set_title(
    "Distribution cumulée (CDF) des séries journalières — Tricastin 1990–2026",
    fontsize=13, fontweight="bold", pad=12
)
ax_cdf.set_yticks(range(0, 101, 10))
ax_cdf.grid(alpha=0.3)
ax_cdf.legend(fontsize=10, loc="upper left")
fig_cdf.tight_layout()
fig_cdf.savefig(FIG_CDF, dpi=150, bbox_inches="tight")
plt.close(fig_cdf)
print(f"  Figure : {FIG_CDF}")

md.h2("8. Distributions cumulées (CDF) – séries journalières")
md.img("CDF des séries journalières", FIG_CDF)
md.blank()


md.hr()
md.text(f"*Fin du rapport – {RAPPORT}*")
md.save()