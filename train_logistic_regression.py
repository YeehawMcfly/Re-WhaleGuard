"""
==========================================================================
 WhaleGuard — Phase 4a: Logistic Regression Baseline Model
==========================================================================
 Trains a Logistic Regression classifier on the fully-engineered NARW
 habitat dataset using a temporally-ordered train/test split.

 Design Decisions:
   - Temporal Split (80/20): Proves the model generalises to *future*
     conditions, not just interpolates known dates.
   - StandardScaler: Logistic regression is sensitive to feature scale,
     so all features are standardised (mean=0, std=1).
   - Class Weighting ("balanced"): Automatically adjusts weights inversely
     proportional to class frequencies, compensating the 1:4 imbalance
     without manual tuning.
   - Imputation (median): Unlike XGBoost, logistic regression cannot
     handle NaN natively; median imputation is robust to outliers.
   - Features EXCLUDE Lat, Lon, Date: Forces the model to learn ocean
     physics and seasonality, not memorise patrol coordinates.
   - L2 Regularisation (default): Reduces overfitting while keeping
     all features in the model for interpretability.

 Outputs:
   - Terminal classification report (Accuracy, Precision, Recall, F1, AUC)
   - images/lr_roc_curve.png
   - images/lr_coefficients.png
   - models/lr_narw_sdm.joblib  (serialised model for deployment)
==========================================================================
"""

import time
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for plot saving
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
import joblib

# =========================================================================
#  Configuration
# =========================================================================

DATA_PATH   = Path("data/processed/ML_Whale_Dataset_Final.csv")
IMG_DIR     = Path("images")
MODEL_DIR   = Path("models")
IMG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Columns to drop from features (model must learn physics, not geography)
DROP_COLS = ["Date", "Lat", "Lon", "Presence"]

# Temporal split ratio
TRAIN_RATIO = 0.80

# Logistic Regression hyperparameters
LR_PARAMS = {
    "C":              1.0,          # inverse regularisation strength
    "penalty":        "l2",         # ridge regularisation
    "solver":         "lbfgs",      # efficient for small-to-medium datasets
    "max_iter":       1000,         # ensure convergence
    "class_weight":   "balanced",   # auto-compensate class imbalance
    "random_state":   42,
}

# Plot style (matching EDA dark theme from XGBoost script)
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
    print_header("WhaleGuard — Logistic Regression NARW Habitat Model")

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

    # Report missing values
    null_counts = features.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"\n  Missing values (will be median-imputed):")
        for col, count in null_cols.items():
            pct = count / len(features) * 100
            print(f"    {col:20s}  {count:,} ({pct:.1f}%)")
    else:
        print(f"  No missing values detected.")

    # ── 3. Temporal Split ────────────────────────────────────────────
    print_section("3. Temporal Train/Test Split")

    # Sort chronologically
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
    print_section("4. Logistic Regression Training")

    # Build a pipeline: Impute NaNs → Scale features → Logistic Regression
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(**LR_PARAMS)),
    ])

    print(f"  Pipeline:")
    print(f"    1. SimpleImputer (strategy=median)")
    print(f"    2. StandardScaler (mean=0, std=1)")
    print(f"    3. LogisticRegression")
    print(f"  Regularisation:   L2 (Ridge), C={LR_PARAMS['C']}")
    print(f"  Solver:           {LR_PARAMS['solver']}")
    print(f"  Class weighting:  balanced (auto-compensates imbalance)")
    print(f"  Max iterations:   {LR_PARAMS['max_iter']}")
    print(f"\n  Training...", end="", flush=True)

    t_train = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t_train
    print(f" done in {train_time:.2f}s")

    # Check convergence
    lr_model = pipeline.named_steps["clf"]
    print(f"  Converged:        {lr_model.n_iter_[0]} iterations")

    # ── 5. Evaluation ────────────────────────────────────────────────
    print_section("5. Model Evaluation (Temporal Test Set)")

    y_pred       = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted 0    Predicted 1")
    print(f"    Actual 0 (Abs)   {tn:>8,}       {fp:>8,}")
    print(f"    Actual 1 (Pres)  {fn:>8,}       {tp:>8,}")
    print(f"\n  Additional Metrics:")
    print(f"    Specificity (TNR):  {specificity:.4f}")
    print(f"    NPV:                {npv:.4f}")

    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Absence", "Presence"]))

    # ── 6. ROC Curve ─────────────────────────────────────────────────
    print_section("6. Generating ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color="#79c0ff")
    ax.plot(fpr, tpr, color="#79c0ff", linewidth=2.5,
            label=f"Logistic Regression (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#f85149", linestyle="--",
            linewidth=1.5, alpha=0.7, label="Random Classifier (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curve — Logistic Regression NARW Habitat Model\n"
                 f"Temporal Validation: Train ≤ {dates.iloc[split_idx-1].date()}, "
                 f"Test ≥ {split_date.date()}",
                 fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    roc_path = IMG_DIR / "lr_roc_curve.png"
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {roc_path}")

    # ── 7. Coefficient Analysis ──────────────────────────────────────
    print_section("7. Coefficient Analysis")

    coefficients = lr_model.coef_[0]
    intercept    = lr_model.intercept_[0]

    # Sort by absolute value for plotting
    abs_sorted_idx = np.argsort(np.abs(coefficients))
    sorted_coeffs  = coefficients[abs_sorted_idx]
    sorted_names   = [feature_names[i] for i in abs_sorted_idx]

    # Colour positive coefficients green (increase whale presence probability)
    # and negative coefficients red (decrease whale presence probability)
    bar_colors = ["#3fb950" if c > 0 else "#f85149" for c in sorted_coeffs]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        range(len(sorted_coeffs)),
        sorted_coeffs,
        color=bar_colors,
        edgecolor="#30363d",
        linewidth=0.8,
        height=0.7,
    )

    # Value labels
    for bar, val in zip(bars, sorted_coeffs):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=11,
                color="#c9d1d9", fontweight="bold")

    ax.axvline(x=0, color="#8b949e", linewidth=0.8, linestyle="-")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=12)
    ax.set_xlabel("Coefficient (Standardised)", fontsize=13)
    ax.set_title("Logistic Regression Coefficients — NARW Habitat Model\n"
                 "Green = increases whale presence probability  |  "
                 "Red = decreases",
                 fontsize=13, pad=15)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    coef_path = IMG_DIR / "lr_coefficients.png"
    fig.savefig(coef_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {coef_path}")

    # Print coefficient ranking
    print(f"\n  Intercept: {intercept:+.4f}")
    print(f"\n  Coefficient Ranking (by absolute magnitude):")
    for rank, idx in enumerate(reversed(abs_sorted_idx)):
        coef = coefficients[idx]
        direction = "↑ whale" if coef > 0 else "↓ whale"
        bar = "█" * int(abs(coef) * 15)
        print(f"    {rank+1}. {feature_names[idx]:20s}  {coef:+.4f}  {direction}  {bar}")

    # ── 8. Odds Ratios ───────────────────────────────────────────────
    print_section("8. Odds Ratios (Interpretability)")

    print(f"\n  Odds Ratios (exp(coefficient)):")
    print(f"  A 1-std increase in the feature multiplies the odds by this factor.\n")
    print(f"    {'Feature':20s}  {'Coeff':>8s}  {'Odds Ratio':>11s}  {'Interpretation'}")
    print(f"    {'─'*20}  {'─'*8}  {'─'*11}  {'─'*30}")

    for idx in reversed(abs_sorted_idx):
        coef = coefficients[idx]
        odds = np.exp(coef)
        if odds > 1:
            pct_change = (odds - 1) * 100
            interp = f"+{pct_change:.1f}% odds per 1σ increase"
        else:
            pct_change = (1 - odds) * 100
            interp = f"-{pct_change:.1f}% odds per 1σ increase"
        print(f"    {feature_names[idx]:20s}  {coef:>+8.4f}  {odds:>11.4f}  {interp}")

    # ── 9. Save Model ────────────────────────────────────────────────
    print_section("9. Saving Model")

    model_path = MODEL_DIR / "lr_narw_sdm.joblib"
    joblib.dump(pipeline, str(model_path))
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Model format: joblib Pipeline (imputer + scaler + classifier)")
    print(f"  ✓ To load: pipeline = joblib.load('{model_path}')")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print_header("Training Complete")
    print(f"  Model Performance:")
    print(f"    ROC-AUC:    {auc:.4f}")
    print(f"    F1-Score:   {f1:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"    Accuracy:   {acc:.4f}")
    print(f"\n  Artifacts:")
    print(f"    {roc_path}")
    print(f"    {coef_path}")
    print(f"    {model_path}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("═" * 68)


if __name__ == "__main__":
    main()
