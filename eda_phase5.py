"""
==========================================================================
 WhaleGuard — Phase 5 EDA: New Spatial Features + Friend's EDA Merge
==========================================================================
 Generates new EDA cells for the 3 spatial features added in Phase 5,
 plus adapts the best ideas from the new-eda branch (presence rate plots,
 month-stratified heatmaps, Kernel PCA).

 Saves all plots to images/ directory.
==========================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ─── Configuration ───────────────────────────────────────────────────────
DATA_PATH = "data/processed/ML_Whale_Dataset_Final.csv"
IMG_DIR   = "images"

# Plot style (matching existing dark theme)
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
})

PRESENCE_COLOR = "#3fb950"
ABSENCE_COLOR  = "#f85149"

# ─── Load Data ───────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df_pres = df[df["Presence"] == 1]
df_abs  = df[df["Presence"] == 0]
print(f"  {len(df):,} rows loaded ({len(df_pres):,} presence, {len(df_abs):,} absence)")

# =========================================================================
#  1. KDE Plots for New Features
# =========================================================================

new_features = {
    "Bathy_Slope": {
        "unit": "m/km",
        "title": "Bathymetric Slope",
        "desc": "Depth gradient magnitude — marks continental shelf break & canyons",
        "xlim": (0, 100),
    },
    "Dist_to_Shore_km": {
        "unit": "km",
        "title": "Distance to Shore",
        "desc": "Haversine distance to nearest coastline",
        "xlim": (0, 350),
    },
    "Dist_to_Shelf_km": {
        "unit": "km",
        "title": "Distance to Shelf Break (200m Isobath)",
        "desc": "Haversine distance to nearest 200m isobath",
        "xlim": (0, 300),
    },
}

print("\n─── Generating KDE Plots for New Features ─────────────────────")
for feat, meta in new_features.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    pres_vals = df_pres[feat].dropna()
    abs_vals  = df_abs[feat].dropna()

    ax.hist(abs_vals, bins=80, density=True, alpha=0.3, color=ABSENCE_COLOR,
            label=f"Absence (n={len(abs_vals):,})")
    ax.hist(pres_vals, bins=80, density=True, alpha=0.3, color=PRESENCE_COLOR,
            label=f"Presence (n={len(pres_vals):,})")

    # KDE overlay
    try:
        pres_vals.plot.kde(ax=ax, color=PRESENCE_COLOR, linewidth=2.5, label="_nolegend_")
        abs_vals.plot.kde(ax=ax, color=ABSENCE_COLOR, linewidth=2.5, label="_nolegend_")
    except Exception:
        pass

    # Mann-Whitney U test
    stat, pval = mannwhitneyu(pres_vals, abs_vals, alternative="two-sided")
    sig_text = f"p = {pval:.2e}" if pval < 0.001 else f"p = {pval:.4f}"

    ax.set_xlabel(f"{meta['title']} ({meta['unit']})", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"KDE — {meta['title']}\n"
                 f"{meta['desc']}\n"
                 f"Mann-Whitney U: {sig_text}",
                 fontsize=13, pad=15)
    ax.legend(fontsize=11, framealpha=0.9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_xlim(meta["xlim"])
    ax.grid(alpha=0.2)

    fig.tight_layout()
    path = f"{IMG_DIR}/kde_{feat.lower()}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# =========================================================================
#  2. Updated Correlation Heatmap (all 10 features)
# =========================================================================

print("\n─── Generating Updated Correlation Heatmap ─────────────────────")

corr_cols = ["SST", "Chlorophyll", "Salinity", "Bathymetry",
             "SST_Gradient", "Is_Thermal_Front", "Month",
             "Bathy_Slope", "Dist_to_Shore_km", "Dist_to_Shelf_km"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="#30363d",
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax,
)
ax.set_title("Feature Correlation Matrix (10 Features)\n"
             "Checking for multicollinearity before model training",
             fontsize=14, pad=15)
fig.tight_layout()
path = f"{IMG_DIR}/correlation_heatmap_full.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {path}")


# =========================================================================
#  3. Presence Rate vs Environmental Variable (Binned) — from new-eda
# =========================================================================

print("\n─── Generating Presence Rate vs. Environmental Bins ─────────────")

features_binned = ["SST", "Chlorophyll", "Salinity", "Bathymetry",
                   "Bathy_Slope", "Dist_to_Shore_km", "Dist_to_Shelf_km",
                   "SST_Gradient"]

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for i, feat in enumerate(features_binned):
    ax = axes[i]
    df_plot = df[[feat, "Presence"]].dropna().copy()

    if feat == "Bathymetry":
        df_plot["_bin"] = pd.qcut(df_plot[feat].abs(), q=10, duplicates="drop")
    else:
        df_plot["_bin"] = pd.qcut(df_plot[feat], q=10, duplicates="drop")

    rates = df_plot.groupby("_bin")["Presence"].mean()

    bars = ax.bar(range(len(rates)), rates.values, color="#58a6ff",
                  edgecolor="#30363d", alpha=0.85)

    # Highlight the peak bin
    peak_idx = np.argmax(rates.values)
    bars[peak_idx].set_color(PRESENCE_COLOR)
    bars[peak_idx].set_edgecolor("white")

    ax.set_xticks(range(len(rates)))
    labels = [str(iv) for iv in rates.index]
    # Shorten interval labels
    short_labels = []
    for lbl in labels:
        try:
            parts = lbl.strip("()[]").split(",")
            short_labels.append(f"{float(parts[0]):.0f}-{float(parts[1]):.0f}")
        except Exception:
            short_labels.append(lbl[:8])
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Presence Rate", fontsize=11)
    ax.set_title(feat, fontsize=13)
    ax.axhline(y=0.2, color=ABSENCE_COLOR, linestyle="--", alpha=0.5,
               linewidth=1, label="Expected (1:4 ratio)")
    ax.grid(axis="y", alpha=0.2)

fig.suptitle("Presence Rate by Environmental Variable Decile\n"
             "Green bar = peak presence rate (optimal habitat zone)",
             fontsize=15, y=1.01)
fig.tight_layout()
path = f"{IMG_DIR}/presence_rate_vs_env.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {path}")


# =========================================================================
#  4. Month-Stratified Presence Rate Heatmaps — from new-eda
# =========================================================================

print("\n─── Generating Month-Stratified Presence Heatmaps ──────────────")

features_heatmap = ["SST", "Chlorophyll", "Salinity", "Bathymetry",
                    "Dist_to_Shore_km", "Dist_to_Shelf_km"]

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
axes = axes.flatten()

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for i, feat in enumerate(features_heatmap):
    ax = axes[i]
    df_plot = df[["Month", feat, "Presence"]].dropna().copy()

    if feat == "Bathymetry":
        df_plot["_bin"] = pd.qcut(df_plot[feat].abs(), q=8, duplicates="drop")
    else:
        df_plot["_bin"] = pd.qcut(df_plot[feat], q=8, duplicates="drop")

    pivot = df_plot.groupby(["Month", "_bin"])["Presence"].mean().unstack()

    sns.heatmap(
        pivot,
        cmap="YlGn",
        annot=False,
        ax=ax,
        cbar_kws={"label": "Presence Rate"},
        linewidths=0.3,
        linecolor="#30363d",
    )

    ax.set_yticklabels([month_names[int(m)-1] for m in pivot.index],
                       rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title(feat, fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Month")

fig.suptitle("Month-Stratified Presence Rate Heatmaps\n"
             "How optimal habitat zones shift seasonally",
             fontsize=15, y=1.01)
fig.tight_layout()
path = f"{IMG_DIR}/month_stratified_presence_heatmaps.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {path}")


# =========================================================================
#  5. Mann-Whitney U Tests for ALL Features (Updated)
# =========================================================================

print("\n─── Mann-Whitney U Tests (All 10 Features) ─────────────────────")

all_features = ["SST", "Chlorophyll", "Salinity", "Bathymetry",
                "SST_Gradient", "Is_Thermal_Front", "Month",
                "Bathy_Slope", "Dist_to_Shore_km", "Dist_to_Shelf_km"]

results = []
for feat in all_features:
    pres_vals = df_pres[feat].dropna()
    abs_vals  = df_abs[feat].dropna()
    stat, pval = mannwhitneyu(pres_vals, abs_vals, alternative="two-sided")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(pres_vals), len(abs_vals)
    r = 1 - (2 * stat) / (n1 * n2)

    results.append({
        "Feature": feat,
        "U-statistic": stat,
        "p-value": pval,
        "Effect Size (r)": abs(r),
        "Significant": "✓" if pval < 0.001 else "✗",
        "Pres. Mean": pres_vals.mean(),
        "Abs. Mean": abs_vals.mean(),
    })

results_df = pd.DataFrame(results).sort_values("Effect Size (r)", ascending=False)
print(results_df.to_string(index=False))

# Save as figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

table = ax.table(
    cellText=[[r["Feature"],
               f"{r['U-statistic']:,.0f}",
               f"{r['p-value']:.2e}",
               f"{r['Effect Size (r)']:.4f}",
               r["Significant"],
               f"{r['Pres. Mean']:.3f}",
               f"{r['Abs. Mean']:.3f}"]
              for _, r in results_df.iterrows()],
    colLabels=["Feature", "U-statistic", "p-value", "Effect Size |r|",
               "Sig.", "Pres. Mean", "Abs. Mean"],
    cellLoc="center",
    loc="center",
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for key, cell in table.get_celld().items():
    cell.set_edgecolor("#30363d")
    cell.set_facecolor("#161b22")
    cell.set_text_props(color="#c9d1d9")
    if key[0] == 0:
        cell.set_facecolor("#21262d")
        cell.set_text_props(color="#c9d1d9", fontweight="bold")

ax.set_title("Mann-Whitney U Tests — All Features\n"
             "Statistical significance of environmental differences "
             "between NARW presence vs. absence",
             fontsize=14, pad=20, color="#c9d1d9")

fig.tight_layout()
path = f"{IMG_DIR}/mann_whitney_all_features.png"
fig.savefig(path, dpi=200, bbox_inches="tight",
            facecolor="#0d1117", edgecolor="none")
plt.close(fig)
print(f"\n  ✓ {path}")


# =========================================================================
#  6. Kernel PCA — from new-eda (updated with new features)
# =========================================================================

print("\n─── Generating Kernel PCA Visualization ────────────────────────")

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

kpca_features = ["SST", "Chlorophyll", "Salinity", "Bathymetry",
                 "SST_Gradient", "Month", "Bathy_Slope",
                 "Dist_to_Shore_km", "Dist_to_Shelf_km"]
df_kpca = df[kpca_features + ["Presence"]].dropna().copy()

print(f"  Kernel PCA input: {len(df_kpca):,} rows (after dropping NaN)")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_kpca[kpca_features])

# Subsample for speed (KernelPCA is O(n²))
np.random.seed(42)
n_sample = min(10000, len(X_scaled))
idx = np.random.choice(len(X_scaled), n_sample, replace=False)
X_sub = X_scaled[idx]
y_sub = df_kpca["Presence"].values[idx]

print(f"  Subsampled to {n_sample:,} for KernelPCA...")

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1, random_state=42)
X_proj = kpca.fit_transform(X_sub)

fig, ax = plt.subplots(figsize=(10, 8))

scatter_abs = ax.scatter(
    X_proj[y_sub == 0, 0], X_proj[y_sub == 0, 1],
    c=ABSENCE_COLOR, alpha=0.15, s=8, label="Absence", rasterized=True,
)
scatter_pres = ax.scatter(
    X_proj[y_sub == 1, 0], X_proj[y_sub == 1, 1],
    c=PRESENCE_COLOR, alpha=0.4, s=12, label="Presence", rasterized=True,
)

ax.set_xlabel("Kernel PC 1", fontsize=13)
ax.set_ylabel("Kernel PC 2", fontsize=13)
ax.set_title("Kernel PCA — Non-Linear Feature Space (RBF Kernel)\n"
             "Clustering indicates XGBoost-learnable environmental niches",
             fontsize=13, pad=15)
ax.legend(fontsize=12, framealpha=0.9, markerscale=3,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.grid(alpha=0.15)

fig.tight_layout()
path = f"{IMG_DIR}/kernel_pca.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {path}")


# =========================================================================
#  7. Presence Rate by Year × Month Heatmap — from new-eda
# =========================================================================

print("\n─── Generating Presence Rate Year×Month Heatmap ─────────────────")

df["Year"] = df["Date"].dt.year

heatmap_data = df.groupby(["Year", "Month"])["Presence"].mean().unstack()

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    linewidths=0.3,
    linecolor="#30363d",
    ax=ax,
    cbar_kws={"label": "Presence Rate"},
    xticklabels=[month_names[i] for i in range(12)],
)
ax.set_title("Presence Rate by Year × Month\n"
             "Temporal distribution of whale sightings across the study period",
             fontsize=14, pad=15)
ax.set_ylabel("Year")
ax.set_xlabel("Month")

fig.tight_layout()
path = f"{IMG_DIR}/presence_rate_year_month.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {path}")


# =========================================================================
#  Summary
# =========================================================================

print("\n" + "═" * 66)
print("  EDA Update Complete!")
print("  New plots saved to images/:")
print("    • kde_bathy_slope.png")
print("    • kde_dist_to_shore_km.png")
print("    • kde_dist_to_shelf_km.png")
print("    • correlation_heatmap_full.png")
print("    • presence_rate_vs_env.png")
print("    • month_stratified_presence_heatmaps.png")
print("    • mann_whitney_all_features.png")
print("    • kernel_pca.png")
print("    • presence_rate_year_month.png")
print("═" * 66)
