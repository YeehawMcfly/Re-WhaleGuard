"""
==========================================================================
 WhaleGuard — Manual Inference Test
==========================================================================
 Tests the trained XGBoost model with hand-crafted scenarios to verify
 ecological plausibility.

 The model expects 10 features in this exact order:
   SST, Chlorophyll, Salinity, Bathymetry, SST_Gradient,
   Is_Thermal_Front, Month, Bathy_Slope, Dist_to_Shore_km, Dist_to_Shelf_km

 The optimal classification threshold (for ≥80% recall) is loaded
 from models/optimal_threshold.txt.
==========================================================================
"""

import pandas as pd
import xgboost as xgb
from pathlib import Path

# ── Load Model & Threshold ───────────────────────────────────────────────
model = xgb.XGBClassifier()
model.load_model("models/xgb_narw_sdm.json")

# Load optimised threshold
threshold = 0.50  # default fallback
threshold_file = Path("models/optimal_threshold.txt")
if threshold_file.exists():
    for line in threshold_file.read_text().splitlines():
        if line.startswith("threshold="):
            threshold = float(line.split("=")[1])
            break

print(f"  Model loaded: models/xgb_narw_sdm.json")
print(f"  Optimal threshold: {threshold:.4f}")

# ── Feature order must match training ────────────────────────────────────
FEATURE_COLS = [
    "SST", "Chlorophyll", "Salinity", "Bathymetry",
    "SST_Gradient", "Is_Thermal_Front", "Month",
    "Bathy_Slope", "Dist_to_Shore_km", "Dist_to_Shelf_km",
]

# ── Test Scenarios ───────────────────────────────────────────────────────
scenarios = [
    {
        "name": "🐳 Cape Cod Bay — Spring Feeding (HIGH expected)",
        "desc": "Shallow shelf, close to shore, cold productive water with fronts",
        "data": {
            "SST": 8.0,                # Cool spring temps
            "Chlorophyll": 6.0,        # High spring bloom
            "Salinity": 32.0,          # Shelf mixing zone
            "Bathymetry": -60.0,       # Shallow shelf (typical CCB)
            "SST_Gradient": 0.06,      # Active thermal front
            "Is_Thermal_Front": 1,     # True
            "Month": 4,               # April (peak CCB feeding)
            "Bathy_Slope": 2.0,        # Gentle shelf slope
            "Dist_to_Shore_km": 8.0,   # Very close to shore
            "Dist_to_Shelf_km": 45.0,  # Inner shelf
        },
    },
    {
        "name": "🐳 Bay of Fundy — Summer Foraging (HIGH expected)",
        "desc": "Classic summer habitat — Calanus aggregation zone",
        "data": {
            "SST": 12.0,              # Summer surface temp
            "Chlorophyll": 4.0,       # Productive
            "Salinity": 31.5,         # Bay of Fundy
            "Bathymetry": -170.0,     # Deep basin
            "SST_Gradient": 0.04,     # Moderate gradient
            "Is_Thermal_Front": 1,    # True
            "Month": 7,              # July
            "Bathy_Slope": 4.0,       # Moderate slope
            "Dist_to_Shore_km": 30.0, # Nearshore
            "Dist_to_Shelf_km": 5.0,  # Near shelf edge
        },
    },
    {
        "name": "❌ Deep Ocean — Mid-Atlantic Ridge (LOW expected)",
        "desc": "Deep pelagic water, far from shore, no prey aggregation",
        "data": {
            "SST": 22.0,              # Warm Gulf Stream
            "Chlorophyll": 0.1,       # Oligotrophic
            "Salinity": 36.5,         # Open ocean
            "Bathymetry": -4500.0,    # Abyssal depth
            "SST_Gradient": 0.01,     # No fronts
            "Is_Thermal_Front": 0,    # False
            "Month": 6,              # June
            "Bathy_Slope": 50.0,      # Steep mid-ocean ridge
            "Dist_to_Shore_km": 400.0,# Far offshore
            "Dist_to_Shelf_km": 350.0,# Far from shelf
        },
    },
    {
        "name": "❌ Tropical Shallow — Florida Keys (LOW expected)",
        "desc": "Too warm, wrong salinity, calving mothers only in winter",
        "data": {
            "SST": 28.0,              # Tropical
            "Chlorophyll": 0.3,       # Low productivity
            "Salinity": 35.8,         # Tropical salinity
            "Bathymetry": -15.0,      # Very shallow
            "SST_Gradient": 0.005,    # No fronts
            "Is_Thermal_Front": 0,    # False
            "Month": 8,              # August (wrong season for SE US)
            "Bathy_Slope": 0.5,       # Flat
            "Dist_to_Shore_km": 5.0,  # Near shore
            "Dist_to_Shelf_km": 80.0, # Wide shelf
        },
    },
]

# ── Run Predictions ──────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  🐳 NARW HABITAT MODEL — MANUAL INFERENCE TEST")
print("═" * 70)

for scenario in scenarios:
    test_df = pd.DataFrame([scenario["data"]])[FEATURE_COLS]
    proba = model.predict_proba(test_df)[0][1]
    prediction = "WHALE HABITAT ✓" if proba >= threshold else "NOT HABITAT ✗"

    print(f"\n  {scenario['name']}")
    print(f"  {scenario['desc']}")
    print(f"  ────────────────────────────────────────")
    print(f"  Probability:  {proba*100:.1f}%")
    print(f"  Threshold:    {threshold*100:.1f}%")
    print(f"  Prediction:   {prediction}")

print("\n" + "═" * 70)
print(f"  Threshold used: {threshold:.4f} (optimised for ≥80% recall)")
print(f"  A probability ≥ {threshold*100:.1f}% triggers a whale habitat alert.")
print("═" * 70 + "\n")