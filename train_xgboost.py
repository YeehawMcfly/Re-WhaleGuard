"""
==========================================================================
 WhaleGuard — Phase 4: XGBoost Model Training & Evaluation
==========================================================================
 Trains an XGBoost classifier on the fully-engineered NARW habitat dataset
 using a temporally-ordered train/test split.

 Design Decisions:
   - Temporal Split (80/20): Proves the model generalises to *future*
     conditions, not just interpolates known dates. This is critical for
     climate-shift resilience (Ji et al., 2024).
   - scale_pos_weight = 4.0: Mathematically counterbalances the 1:4
     pseudo-absence ratio from Gowan & Ortega-Ortiz (2014).
   - Features EXCLUDE Lat, Lon, Date: Forces the model to learn ocean
     physics and seasonality, not memorise patrol coordinates.
   - XGBoost's Sparsity-Aware Split Finding natively handles the ~1-5%
     NaN values in SST, Chlorophyll, and Salinity (Ji et al., 2024).

 Outputs:
   - Terminal classification report (Accuracy, Precision, Recall, F1, AUC)
   - images/roc_curve.png
   - images/feature_importance.png
   - models/xgb_narw_sdm.json  (serialised model for deployment)
==========================================================================
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# =========================================================================
#  Configuration
# =========================================================================

DATA_PATH   = Path("data/processed/ML_Whale_Dataset_Engineered_Patched.csv")
IMG_DIR     = Path("images")
MODEL_DIR   = Path("models")
IMG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Columns to drop from features (model must learn physics, not geography)
DROP_COLS = ["Date", "Lat", "Lon", "Presence"]

# Temporal split ratio
TRAIN_RATIO = 0.80

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "tree_method":      "hist",
    "random_state":     42,
    "verbosity":        0,
}

# Plot style (matching EDA dark theme)
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


# =========================================================================
#  Helpers
# =========================================================================

def print_header(title: str):
    width = 66
    print("\n" + "╔" + "═" * width + "╗")
    print("║  " + title.ljust(width - 2) + "║")
    print("╚" + "═" * width + "╝\n")


def print_section(title: str):
    print(f"\n─── {title} " + "─" * max(0, 58 - len(title)))


# =========================================================================
#  Main Pipeline
# =========================================================================

def main():
    t_start = time.time()
    print_header("WhaleGuard — XGBoost NARW Habitat Model")

    # ── 1. Load Data ─────────────────────────────────────────────────
    print_section("1. Data Loading")

    if not DATA_PATH.exists():
        print(f"  ✗ File not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    # ── 2. Feature / Target Separation ───────────────────────────────
    print_section("2. Feature Engineering")

    target = df["Presence"]
    features = df.drop(columns=DROP_COLS)

    # Ensure Is_Thermal_Front is numeric (bool → int)
    if "Is_Thermal_Front" in features.columns:
        features["Is_Thermal_Front"] = features["Is_Thermal_Front"].astype(int)

    feature_names = list(features.columns)
    print(f"  Target: Presence (1 = whale, 0 = background)")
    print(f"  Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"  Dropped: {', '.join(DROP_COLS[:-1])} (prevent geographic memorisation)")

    # ── 3. Temporal Split ────────────────────────────────────────────
    print_section("3. Temporal Train/Test Split")

    # Sort chronologically (should already be, but enforce)
    sort_idx = df["Date"].argsort()
    features = features.iloc[sort_idx].reset_index(drop=True)
    target   = target.iloc[sort_idx].reset_index(drop=True)
    dates    = df["Date"].iloc[sort_idx].reset_index(drop=True)

    split_idx  = int(len(df) * TRAIN_RATIO)
    split_date = dates.iloc[split_idx]

    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx],   target.iloc[split_idx:]

    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    ratio = n_neg_train / n_pos_train if n_pos_train > 0 else 4.0

    print(f"  Strategy: Chronological (train on past, test on future)")
    print(f"  Split point: {split_date.date()} ({TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%)")
    print(f"  Train: {len(X_train):,} rows ({dates.iloc[0].date()} → {dates.iloc[split_idx-1].date()})")
    print(f"  Test:  {len(X_test):,} rows  ({split_date.date()} → {dates.iloc[-1].date()})")
    print(f"  Train class balance: {n_pos_train:,} pos / {n_neg_train:,} neg (ratio 1:{ratio:.1f})")

    # ── 4. Model Training ────────────────────────────────────────────
    print_section("4. XGBoost Training")

    # scale_pos_weight counterbalances the 1:4 imbalance
    XGB_PARAMS["scale_pos_weight"] = ratio

    model = xgb.XGBClassifier(**XGB_PARAMS)

    print(f"  Algorithm:        XGBClassifier (gradient-boosted trees)")
    print(f"  n_estimators:     {XGB_PARAMS['n_estimators']}")
    print(f"  max_depth:        {XGB_PARAMS['max_depth']}")
    print(f"  learning_rate:    {XGB_PARAMS['learning_rate']}")
    print(f"  scale_pos_weight: {ratio:.2f} (compensates 1:{ratio:.0f} imbalance)")
    print(f"  eval_metric:      AUC")
    print(f"\n  Training...", end="", flush=True)

    t_train = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_time = time.time() - t_train
    print(f" done in {train_time:.1f}s")

    # ── 5. Evaluation ────────────────────────────────────────────────
    print_section("5. Model Evaluation (Temporal Test Set)")

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_pred_proba)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │          PERFORMANCE METRICS             │")
    print(f"  ├─────────────────────┬───────────────────┤")
    print(f"  │  Accuracy           │  {acc:>14.4f}    │")
    print(f"  │  Precision          │  {precision:>14.4f}    │")
    print(f"  │  Recall (Sens.)     │  {recall:>14.4f}    │")
    print(f"  │  F1-Score           │  {f1:>14.4f}    │")
    print(f"  │  ROC-AUC            │  {auc:>14.4f}    │")
    print(f"  └─────────────────────┴───────────────────┘")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted 0    Predicted 1")
    print(f"    Actual 0 (Abs)   {tn:>8,}       {fp:>8,}")
    print(f"    Actual 1 (Pres)  {fn:>8,}       {tp:>8,}")

    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Absence", "Presence"]))

    # ── 6. ROC Curve ─────────────────────────────────────────────────
    print_section("6. Generating ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color="#58a6ff")
    ax.plot(fpr, tpr, color="#58a6ff", linewidth=2.5,
            label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#f85149", linestyle="--",
            linewidth=1.5, alpha=0.7, label="Random Classifier (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curve — NARW Habitat Model\n"
                 f"Temporal Validation: Train ≤ {dates.iloc[split_idx-1].date()}, "
                 f"Test ≥ {split_date.date()}",
                 fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    roc_path = IMG_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {roc_path}")

    # ── 7. Feature Importance ────────────────────────────────────────
    print_section("7. Generating Feature Importance Chart")

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.cividis(np.linspace(0.2, 0.9, len(sorted_idx)))
    bars = ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color=colors,
        edgecolor="#30363d",
        linewidth=0.8,
        height=0.7,
    )

    # Value labels
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, color="#c9d1d9",
                fontweight="bold")

    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=12)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
    ax.set_title("XGBoost Feature Importance — NARW Habitat Model\n"
                 "Which ocean variables drive whale presence predictions?",
                 fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fi_path = IMG_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {fi_path}")

    # Print ranking
    print(f"\n  Feature Importance Ranking:")
    for rank, idx in enumerate(reversed(sorted_idx)):
        bar = "█" * int(importances[idx] * 40)
        print(f"    {rank+1}. {feature_names[idx]:20s}  {importances[idx]:.4f}  {bar}")

    # ── 8. Save Model ────────────────────────────────────────────────
    print_section("8. Saving Model")

    model_path = MODEL_DIR / "xgb_narw_sdm.json"
    model.save_model(str(model_path))
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Model format: XGBoost JSON (portable, version-safe)")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print_header("Training Complete")
    print(f"  Model Performance:")
    print(f"    ROC-AUC:    {auc:.4f}")
    print(f"    F1-Score:   {f1:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"\n  Artifacts:")
    print(f"    {roc_path}")
    print(f"    {fi_path}")
    print(f"    {model_path}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("═" * 68)


if __name__ == "__main__":
    main()
