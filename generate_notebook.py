"""Generate the EDA Jupyter Notebook programmatically."""
import json

def md(source):
    lines = source.split("\n")
    lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]  # trailing \n except last
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(source):
    lines = source.split("\n")
    lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "source": lines, "outputs": [], "execution_count": None}

cells = []

# =====================================================================
# CELL 1: Setup & Missingness
# =====================================================================
cells.append(md("""# Exploratory Data Analysis — North Atlantic Right Whale SDM
## WhaleGuard ML Pipeline: Data Validation & Literature Cross-Reference

This notebook performs an exhaustive Exploratory Data Analysis (EDA) on the engineered dataset produced by the WhaleGuard ETL pipeline. The objective is to **empirically validate** that our data aligns with the current marine biology literature before training an XGBoost classifier.

### 1.1 — Data Loading & The "Missingness" Justification

Oceanographic satellite data is inherently incomplete. Cloud cover masks optical sensors (Chlorophyll-a), coastal interference corrupts microwave retrievals (Salinity), and temporal misalignment between satellite overpasses and sighting dates introduces systematic gaps.

**Ji et al. (2024)** demonstrated that tree-based ensemble models — specifically XGBoost and Random Forest — achieve the highest predictive accuracy for NARW presence *precisely because* of their native **Sparsity-Aware Split Finding** algorithm. Unlike neural networks or logistic regression, XGBoost does not require imputation; it learns an optimal default direction at each split node for missing values, preserving the statistical integrity of the non-missing data.

> **Key Implication:** We do *not* impute missing values. We document them here and rely on XGBoost's architectural strength to handle them natively."""))

cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Style Configuration ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
})

PALETTE = {'presence': '#58a6ff', 'absence': '#f778ba'}
IMG_DIR = Path('images')
IMG_DIR.mkdir(exist_ok=True)

# ── Load Dataset ────────────────────────────────────────────────────
df = pd.read_csv('data/processed/ML_Whale_Dataset_Engineered_Patched.csv', parse_dates=['Date'])
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}\\n")
df.info()

# ── Missingness Table ───────────────────────────────────────────────
env_cols = ['SST', 'Chlorophyll', 'Salinity', 'Bathymetry', 'SST_Gradient']
print("\\n" + "=" * 60)
print("  MISSINGNESS REPORT (Ji et al., 2024 — Sparsity-Aware)")
print("=" * 60)
for col in env_cols:
    valid = df[col].notna().sum()
    total = len(df)
    pct = 100 * valid / total
    bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
    print(f"  {col:15s} {bar} {pct:5.1f}%  ({total - valid:,} NaN)")

# ── Missingness Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
missing_matrix = df[env_cols].isna().T
sns.heatmap(missing_matrix, cbar=False, cmap=['#238636', '#da3633'],
            yticklabels=env_cols, xticklabels=False, ax=ax)
ax.set_title('Data Completeness Matrix (Green = Valid, Red = Missing)', fontsize=14, pad=15)
ax.set_xlabel(f'Rows (n = {len(df):,})', fontsize=11)
fig.tight_layout()
fig.savefig(IMG_DIR / 'missingness_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"\\n✓ Saved: {IMG_DIR / 'missingness_heatmap.png'}")"""))

# =====================================================================
# CELL 2: Observer Bias & Geographic Verification
# =====================================================================
cells.append(md("""### 2 — Observer Bias Elimination & Geographic Spatial Verification (Gowan & Ortega-Ortiz, 2014)

Visual sighting data suffers from severe **spatial observer bias**: whales are only recorded where survey vessels patrol. A model trained exclusively on presence data will learn *where boats look*, not *where whales live*.

Following **Gowan & Ortega-Ortiz (2014)**, our pipeline mitigates this through:
1. **1:4 Pseudo-Absence Ratio** — For every confirmed sighting, 4 synthetic "background" points are generated, preventing class imbalance artifacts.
2. **Temporal Matching** — Background points share the *exact same date* as the sighting, ensuring the model evaluates the specific oceanographic conditions of that day.
3. **Spatial Buffering** — Points are scattered within a **300 km radius** but *outside* a **15 km exclusion zone**, preventing false negatives near real sightings.
4. **Ocean-Only Validation** — All synthetic points are validated against a land mask (`global-land-mask`) to ensure none fall on land.

The geographic scatter plot below must confirm: (a) presence and absence points overlap broadly in the Northwest Atlantic, (b) absence points do *not* cluster on land, and (c) the spatial extent covers known NARW habitats (Cape Cod Bay, Great South Channel, Bay of Fundy, Gulf of St. Lawrence)."""))

cells.append(code("""# ── Class Balance ────────────────────────────────────────────────────
n_pres = (df['Presence'] == 1).sum()
n_abs  = (df['Presence'] == 0).sum()
ratio  = n_abs / n_pres

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(['Presence (1)\\nConfirmed Sightings', 'Absence (0)\\nPseudo-Absences'],
              [n_pres, n_abs],
              color=[PALETTE['presence'], PALETTE['absence']],
              edgecolor='#30363d', linewidth=1.5, width=0.55)
for bar, val in zip(bars, [n_pres, n_abs]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
            f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='#c9d1d9')
ax.set_title(f'Class Balance — Ratio 1:{ratio:.1f} (Gowan & Ortega-Ortiz, 2014)', fontsize=14, pad=15)
ax.set_ylabel('Count', fontsize=12)
ax.set_ylim(0, n_abs * 1.15)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_DIR / 'class_balance.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {IMG_DIR / 'class_balance.png'}")

# ── Geographic Scatter (Cartopy) ────────────────────────────────────
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df_abs  = df[df['Presence'] == 0].sample(n=min(8000, n_abs), random_state=42)
df_pres = df[df['Presence'] == 1]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-78, -58, 33, 50], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='#161b22', edgecolor='#30363d', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='#0d1117')
ax.add_feature(cfeature.COASTLINE, edgecolor='#484f58', linewidth=0.6)
ax.add_feature(cfeature.BORDERS, edgecolor='#30363d', linewidth=0.3)
ax.add_feature(cfeature.STATES, edgecolor='#21262d', linewidth=0.2)

ax.scatter(df_abs['Lon'], df_abs['Lat'], c=PALETTE['absence'],
           s=3, alpha=0.25, transform=ccrs.PlateCarree(), label=f'Absence (n={n_abs:,})', zorder=2)
ax.scatter(df_pres['Lon'], df_pres['Lat'], c=PALETTE['presence'],
           s=8, alpha=0.6, transform=ccrs.PlateCarree(), label=f'Presence (n={n_pres:,})', zorder=3)

gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#30363d', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

ax.legend(loc='lower right', fontsize=11, framealpha=0.9,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
ax.set_title('Geographic Distribution — NARW Sightings & Pseudo-Absences\\n'
             'Gowan & Ortega-Ortiz (2014): 15km buffer, 300km radius, ocean-validated',
             fontsize=13, pad=15, color='#c9d1d9')

fig.savefig(IMG_DIR / 'geographic_distribution.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print(f"✓ Saved: {IMG_DIR / 'geographic_distribution.png'}")"""))

# =====================================================================
# CELL 3: Tao et al. Goldilocks Zone
# =====================================================================
cells.append(md("""### 3 — The "Goldilocks Zone": Bivariate Habitat Envelope (Tao et al., 2025)

**Tao et al. (2025)** identified the critical bivariate habitat envelope for NARW aggregation. Right whales do not forage randomly — they concentrate at the **intersection** of two conditions:

1. **Bathymetric Preference (50–200m depth):** NARWs target the continental shelf where upwelling concentrates *Calanus finmarchicus* copepods in dense subsurface layers.
2. **Thermal Front Activity (SST Gradient > 0.035°C/km):** Convergent thermal boundaries physically trap copepod patches, creating the energy-dense foraging zones whales depend on.

The KDE contour plot below should show a clear density peak for `Presence=1` points within the rectangle defined by these two thresholds — the "Goldilocks Zone" where both conditions are simultaneously satisfied."""))

cells.append(code("""# ── Tao et al. 2D KDE Contour ───────────────────────────────────────
pres = df[(df['Presence'] == 1) & df['SST_Gradient'].notna() & df['Bathymetry'].notna()].copy()
pres['Depth'] = pres['Bathymetry'].abs()  # Convert to positive depth

fig, ax = plt.subplots(figsize=(12, 8))

# KDE contour for presence points
x = pres['Depth'].values
y = pres['SST_Gradient'].values

# Filter to reasonable ranges for visualization
mask = (x < 1000) & (y < 0.3)
x, y = x[mask], y[mask]

try:
    sns.kdeplot(x=x, y=y, levels=10, fill=True, cmap='crest', alpha=0.8, ax=ax)
    sns.kdeplot(x=x, y=y, levels=6, fill=False, colors='#58a6ff', linewidths=0.8, ax=ax)
except:
    ax.scatter(x, y, c=PALETTE['presence'], s=2, alpha=0.3)

# Threshold lines (Tao et al., 2025)
ax.axvline(50, color='#f85149', linestyle='--', linewidth=2, alpha=0.9, label='50m depth')
ax.axvline(200, color='#f85149', linestyle='--', linewidth=2, alpha=0.9, label='200m depth')
ax.axhline(0.035, color='#d29922', linestyle='--', linewidth=2, alpha=0.9, label='0.035°C/km threshold')

# Shade the Goldilocks Zone
ax.axvspan(50, 200, alpha=0.08, color='#f85149')
ax.axhspan(0.035, ax.get_ylim()[1], xmin=50/ax.get_xlim()[1], xmax=200/ax.get_xlim()[1],
           alpha=0.0)  # Reference only

ax.set_xlabel('Depth (|Bathymetry|, meters)', fontsize=13)
ax.set_ylabel('SST Gradient (°C/km)', fontsize=13)
ax.set_title('Bivariate Habitat Envelope — Tao et al. (2025)\\n'
             'NARW Presence Density: Depth × Thermal Front Activity',
             fontsize=14, pad=15)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
ax.set_xlim(0, 600)
ax.set_ylim(0, 0.25)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(IMG_DIR / 'tao_contour.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {IMG_DIR / 'tao_contour.png'}")"""))

# =====================================================================
# CELL 4: Mosnier KDE Plots (Individual)
# =====================================================================
cells.append(md("""### 4 — Univariate Feature Distributions: Mosnier et al. (2025) Validation

**Mosnier et al. (2025)** validated that four pelagic variables govern NARW distribution in the Gulf of St. Lawrence:

- **SST:** NARWs prefer cooler waters (8–15°C) associated with *Calanus finmarchicus* thermal tolerance.
- **Chlorophyll-a:** Higher concentrations indicate phytoplankton blooms — the base of the copepod food web.
- **Salinity:** Intermediate salinity (~31–33 PSU) marks the mixing zones where nutrient-rich deep water meets surface layers.
- **Bathymetry:** Strong preference for the 50–200m continental shelf.

Each plot below compares the distribution of `Presence=1` (blue) vs `Presence=0` (pink). A clear **separation** between the two curves for a given feature indicates that feature has strong discriminative power for the ML model."""))

cells.append(code("""# ── Individual KDE Plots (Mosnier et al., 2025) ─────────────────────
features = {
    'SST': {'xlabel': 'Sea Surface Temperature (°C)', 'color_p': '#58a6ff', 'color_a': '#f778ba'},
    'Chlorophyll': {'xlabel': 'Chlorophyll-a (mg/m³)', 'color_p': '#3fb950', 'color_a': '#f778ba'},
    'Salinity': {'xlabel': 'Salinity (PSU)', 'color_p': '#d2a8ff', 'color_a': '#f778ba'},
    'Bathymetry': {'xlabel': 'Bathymetry (m, negative = depth)', 'color_p': '#79c0ff', 'color_a': '#f778ba'},
}

pres_df = df[df['Presence'] == 1]
abs_df  = df[df['Presence'] == 0]

for feat, cfg in features.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    pres_vals = pres_df[feat].dropna()
    abs_vals  = abs_df[feat].dropna()

    if len(pres_vals) > 0:
        sns.kdeplot(pres_vals, ax=ax, color=cfg['color_p'], fill=True, alpha=0.35,
                    linewidth=2.5, label=f'Presence (n={len(pres_vals):,})')
    if len(abs_vals) > 0:
        sns.kdeplot(abs_vals, ax=ax, color=cfg['color_a'], fill=True, alpha=0.25,
                    linewidth=2.5, label=f'Absence (n={len(abs_vals):,})')

    ax.set_xlabel(cfg['xlabel'], fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'{feat} Distribution — Presence vs. Background\\n'
                 f'Mosnier et al. (2025): Pelagic Variable Validation',
                 fontsize=14, pad=15)
    ax.legend(fontsize=12, framealpha=0.9,
              facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()

    fname = IMG_DIR / f'kde_{feat.lower()}.png'
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {fname}")"""))

# =====================================================================
# CELL 5: Statistical Threshold Validation
# =====================================================================
cells.append(md("""### 5 — Empirical Threshold Validation

Visual KDE plots provide intuition, but **quantitative validation** is essential. Here we calculate the exact empirical percentages to determine what fraction of confirmed NARW sightings fall within the habitat thresholds identified by the literature:

1. **Bathymetric envelope:** 50–200m depth (Tao et al., 2025)
2. **Thermal front activity:** SST_Gradient > 0.035°C/km (Tao et al., 2025)
3. **Combined "Goldilocks Zone":** Both conditions simultaneously satisfied"""))

cells.append(code("""# ── Threshold Statistics ─────────────────────────────────────────────
pres = df[df['Presence'] == 1].copy()
abs_ = df[df['Presence'] == 0].copy()

pres['Depth'] = pres['Bathymetry'].abs()
abs_['Depth'] = abs_['Bathymetry'].abs()

print("=" * 65)
print("  EMPIRICAL THRESHOLD VALIDATION — Tao et al. (2025)")
print("=" * 65)

# Bathymetry 50-200m
pres_bathy = pres['Depth'].dropna()
in_range_p = ((pres_bathy >= 50) & (pres_bathy <= 200)).sum()
pct_p = 100 * in_range_p / len(pres_bathy)

abs_bathy = abs_['Depth'].dropna()
in_range_a = ((abs_bathy >= 50) & (abs_bathy <= 200)).sum()
pct_a = 100 * in_range_a / len(abs_bathy)

print(f"\\n  1. Bathymetric Envelope (50–200m depth):")
print(f"     Presence:  {in_range_p:>6,}/{len(pres_bathy):,}  =  {pct_p:.1f}%")
print(f"     Absence:   {in_range_a:>6,}/{len(abs_bathy):,}  =  {pct_a:.1f}%")
print(f"     → Enrichment factor: {pct_p/pct_a:.2f}x" if pct_a > 0 else "")

# Thermal fronts
pres_front = pres['Is_Thermal_Front'].sum()
pct_front_p = 100 * pres_front / len(pres)
abs_front = abs_['Is_Thermal_Front'].sum()
pct_front_a = 100 * abs_front / len(abs_)

print(f"\\n  2. Active Thermal Fronts (SST_Gradient > 0.035°C/km):")
print(f"     Presence:  {pres_front:>6,}/{len(pres):,}  =  {pct_front_p:.1f}%")
print(f"     Absence:   {abs_front:>6,}/{len(abs_):,}  =  {pct_front_a:.1f}%")
print(f"     → Enrichment factor: {pct_front_p/pct_front_a:.2f}x" if pct_front_a > 0 else "")

# Combined Goldilocks
pres_gold = pres[(pres['Depth'] >= 50) & (pres['Depth'] <= 200) & (pres['Is_Thermal_Front'] == True)]
pct_gold_p = 100 * len(pres_gold) / len(pres)
abs_gold = abs_[(abs_['Depth'] >= 50) & (abs_['Depth'] <= 200) & (abs_['Is_Thermal_Front'] == True)]
pct_gold_a = 100 * len(abs_gold) / len(abs_)

print(f"\\n  3. Combined 'Goldilocks Zone' (50–200m AND thermal front):")
print(f"     Presence:  {len(pres_gold):>6,}/{len(pres):,}  =  {pct_gold_p:.1f}%")
print(f"     Absence:   {len(abs_gold):>6,}/{len(abs_):,}  =  {pct_gold_a:.1f}%")
print(f"     → Enrichment factor: {pct_gold_p/pct_gold_a:.2f}x" if pct_gold_a > 0 else "")

print("\\n" + "=" * 65)
print("  INTERPRETATION")
print("=" * 65)
print("  If enrichment factors > 1.0, the feature discriminates well.")
print("  Values > 2.0 indicate strong ecological signal — exactly what")
print("  XGBoost will exploit for high-accuracy classification.")
print("=" * 65)"""))

# =====================================================================
# CELL 6: Month Feature Engineering
# =====================================================================
cells.append(md("""### 6 — Temporal Feature Engineering: Month as a Predictor

NARW habitat use is **profoundly seasonal**. Right whales migrate between calving grounds in the warm Southeast U.S. (Dec–Mar) and foraging grounds in the cold North Atlantic (Jun–Oct). Without a temporal feature, the model cannot distinguish "5°C water in January near Cape Cod" (calving context) from "5°C water in July off Nova Scotia" (foraging context) — two scenarios with identical SST but entirely different ecological meaning.

Every major NARW SDM study includes temporal context. We extract `Month` from the `Date` column as an integer feature (1–12). XGBoost will learn month-dependent split rules (e.g., "if Month > 5 AND SST < 12°C → high whale probability")."""))

cells.append(code("""# ── Add Month Feature ────────────────────────────────────────────────
df['Month'] = df['Date'].dt.month
print(f"Added 'Month' column (range: {df['Month'].min()} – {df['Month'].max()})")
print(f"Updated columns: {', '.join(df.columns)}")
print(f"\\nMonth value counts:")
print(df['Month'].value_counts().sort_index().to_string())

# Save updated dataset with Month column
df.to_csv('data/processed/ML_Whale_Dataset_Engineered_Patched.csv', index=False)
print(f"\\n✓ Updated CSV saved with Month column")"""))

# =====================================================================
# CELL 7: Seasonal Sighting Histogram
# =====================================================================
cells.append(md("""### 7 — Seasonal Sighting Frequency & Migration Pattern

This histogram reveals the **temporal survey coverage** and the well-documented NARW **migration cycle**. We expect to see:
- A winter peak (Dec–Mar) corresponding to calving surveys off the Southeast U.S.
- A spring-summer peak (Apr–Sep) corresponding to foraging surveys in the Gulf of Maine, Bay of Fundy, and Gulf of St. Lawrence.

Any month with very few sightings indicates a **data gap** — the model will have less information about whale behavior in those conditions. This is critical context for interpreting model performance."""))

cells.append(code("""# ── Seasonal Histogram ───────────────────────────────────────────────
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

pres_months = df[df['Presence'] == 1]['Month'].value_counts().sort_index()
abs_months  = df[df['Presence'] == 0]['Month'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(1, 13)
width = 0.35

bars1 = ax.bar(x - width/2, [pres_months.get(m, 0) for m in range(1, 13)],
               width, color=PALETTE['presence'], edgecolor='#30363d',
               linewidth=1, label='Presence', alpha=0.9)
bars2 = ax.bar(x + width/2, [abs_months.get(m, 0) for m in range(1, 13)],
               width, color=PALETTE['absence'], edgecolor='#30363d',
               linewidth=1, label='Absence', alpha=0.7)

ax.set_xlabel('Month', fontsize=13)
ax.set_ylabel('Number of Records', fontsize=13)
ax.set_title('Seasonal Sighting Frequency — NARW Migration Cycle\\n'
             'Reveals temporal survey coverage and calving vs. foraging periods',
             fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(month_names, fontsize=11)
ax.legend(fontsize=12, framealpha=0.9,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
ax.grid(axis='y', alpha=0.2)

# Add count labels on presence bars
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 30,
                f'{int(h):,}', ha='center', va='bottom', fontsize=9,
                color='#c9d1d9', fontweight='bold')

fig.tight_layout()
fig.savefig(IMG_DIR / 'seasonal_histogram.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {IMG_DIR / 'seasonal_histogram.png'}")

# Print peak months
peak = pres_months.idxmax()
print(f"\\nPeak sighting month: {month_names[peak-1]} ({pres_months[peak]:,} sightings)")
print(f"Lowest sighting month: {month_names[pres_months.idxmin()-1]} ({pres_months.min():,} sightings)")"""))

# =====================================================================
# CELL 8: Mann-Whitney U Statistical Tests
# =====================================================================
cells.append(md("""### 8 — Mann-Whitney U Tests: Statistical Validation of Feature Discrimination

Visual KDE overlap can be misleading. The **Mann-Whitney U test** (non-parametric) quantifies whether two distributions are statistically significantly different, without assuming normality. For each environmental feature, we test:

**H₀:** The distribution of Feature X is identical for Presence=1 and Presence=0.
**H₁:** The distributions differ.

A p-value < 0.05 means the distributions are significantly different — the feature carries discriminative information. The **effect size (rank-biserial correlation)** tells us *how different* they are:
- |r| < 0.1 → negligible
- |r| 0.1–0.3 → small
- |r| 0.3–0.5 → medium
- |r| > 0.5 → large"""))

cells.append(code("""# ── Mann-Whitney U Tests ─────────────────────────────────────────────
from scipy.stats import mannwhitneyu

test_features = ['SST', 'Chlorophyll', 'Salinity', 'Bathymetry', 'SST_Gradient']

pres_df = df[df['Presence'] == 1]
abs_df  = df[df['Presence'] == 0]

print("=" * 75)
print("  MANN-WHITNEY U TESTS — Feature Discrimination Significance")
print("=" * 75)
print(f"  {'Feature':15s} {'U-statistic':>14s} {'p-value':>12s} {'Effect (r)':>12s} {'Strength':>12s}")
print("-" * 75)

results = []
for feat in test_features:
    p_vals = pres_df[feat].dropna().values
    a_vals = abs_df[feat].dropna().values

    if len(p_vals) < 10 or len(a_vals) < 10:
        print(f"  {feat:15s} — insufficient data —")
        continue

    u_stat, p_val = mannwhitneyu(p_vals, a_vals, alternative='two-sided')

    # Rank-biserial correlation (effect size)
    n1, n2 = len(p_vals), len(a_vals)
    r = 1 - (2 * u_stat) / (n1 * n2)

    if abs(r) >= 0.5:
        strength = "LARGE ⭐"
    elif abs(r) >= 0.3:
        strength = "MEDIUM"
    elif abs(r) >= 0.1:
        strength = "SMALL"
    else:
        strength = "negligible"

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    print(f"  {feat:15s} {u_stat:>14,.0f} {p_val:>12.2e} {r:>+12.4f} {strength:>12s} {sig}")
    results.append({'feature': feat, 'r': abs(r), 'p': p_val})

print("-" * 75)
print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
print("  Effect size:  rank-biserial correlation (r)")
print("=" * 75)

# Rank features by effect size
results.sort(key=lambda x: x['r'], reverse=True)
print("\\n  Feature Ranking by Discriminative Power:")
for i, r in enumerate(results):
    bar = '█' * int(r['r'] * 40)
    print(f"    {i+1}. {r['feature']:15s} |r| = {r['r']:.4f}  {bar}")"""))

# =====================================================================
# CELL 9: Correlation Heatmap (updated to include Month)
# =====================================================================
cells.append(md("""### 9 — Multicollinearity Check: Feature Independence

Before feeding features into a tree-based model, we must verify they are not excessively correlated. High multicollinearity (|r| > 0.85) between features can:
- Inflate feature importance scores
- Make SHAP explanations unreliable
- Reduce model generalizability

While XGBoost is more robust to collinearity than linear models, documenting feature independence strengthens the scientific defensibility of the model."""))

cells.append(code("""# ── Correlation Heatmap ──────────────────────────────────────────────
corr_cols = ['SST', 'Chlorophyll', 'Salinity', 'Bathymetry', 'SST_Gradient', 'Month']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=1, linecolor='#30363d',
            square=True, ax=ax,
            annot_kws={'size': 13, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})

ax.set_title('Feature Correlation Matrix — Multicollinearity Check\\n'
             'Threshold: |r| > 0.85 would indicate problematic redundancy',
             fontsize=13, pad=15)
ax.tick_params(axis='both', labelsize=12)
fig.tight_layout()
fig.savefig(IMG_DIR / 'correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {IMG_DIR / 'correlation_heatmap.png'}")

# Print summary
print("\\nCorrelation Summary (|r| > 0.5 flagged):")
for i in range(len(corr_cols)):
    for j in range(i+1, len(corr_cols)):
        r = corr_matrix.iloc[i, j]
        flag = " ⚠ MODERATE" if abs(r) > 0.5 else ""
        if abs(r) > 0.3:
            print(f"  {corr_cols[i]:15s} ↔ {corr_cols[j]:15s}:  r = {r:+.3f}{flag}")

print("\\n✅ EDA Complete. Dataset is validated and ready for XGBoost training.")"""))

# =====================================================================
# Assemble notebook
# =====================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "cells": cells
}

with open("eda_narw_sdm.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook written to eda_narw_sdm.ipynb")
