"""
==========================================================================
 WhaleGuard Phase 3 — Data Sanity Check & Thermal Front Engineering
==========================================================================
 Loads the Phase 2 base dataset and:

   Part 1 — Sanity Check
     Validates data balance, completeness, and physical plausibility.

   Part 2 — SST Gradient / Thermal Front Detection
     Whales forage at thermal fronts where SST gradient > 0.035°C/km
     (Tao et al., 2025). For each unique date, fetches the MUR SST
     spatial slab from NOAA ERDDAP and computes the Sobel-style
     gradient magnitude in °C/km, then extracts values at each point.

   Part 3 — Export
     Saves the engineered dataset with new columns:
       • SST_Gradient   — spatial gradient magnitude (°C/km)
       • Is_Thermal_Front — boolean flag (gradient > 0.035°C/km)

 References:
   Tao et al. (2025) — Thermal front threshold methodology
==========================================================================
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message=".*SerializationWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
#  Configuration
# =========================================================================

INPUT_CSV  = Path("data/processed/ML_Whale_Dataset_Base.csv")
OUTPUT_CSV = Path("data/processed/ML_Whale_Dataset_Engineered.csv")

# MUR SST OPeNDAP endpoint (0.01° daily, 2002–present)
SST_URL      = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41"
SST_VARIABLE = "analysed_sst"

# Thermal front threshold (Tao et al., 2025)
FRONT_THRESHOLD_C_PER_KM = 0.035

# MUR SST resolution: 0.01° ≈ 1.11 km at mid-latitudes (~42°N for NARW)
# More precisely: 1° latitude ≈ 111.32 km always
#                 1° longitude ≈ 111.32 * cos(lat) km
PIXEL_SIZE_DEG = 0.01

# Spatial buffer around bounding box for gradient edge effects
SLAB_BUFFER_DEG = 0.5

# Retry settings for OPeNDAP resilience
MAX_RETRIES = 3

# =========================================================================
#  Logging
# =========================================================================

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("WhaleGuard_Phase3")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "phase3.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

log = _setup_logging()


# =========================================================================
#  Part 1 — Data Sanity Check
# =========================================================================

def run_sanity_check(df: pd.DataFrame) -> None:
    """Print a comprehensive sanity check report for the Phase 2 dataset."""

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║           DATA SANITY CHECK — Phase 2 Validation           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    total = len(df)
    env_cols = ["SST", "Chlorophyll", "Salinity", "Bathymetry"]

    # ── 1. Balance ───────────────────────────────────────────────────────
    log.info("─── 1. CLASS BALANCE ────────────────────────────────────────")
    n_present = int(df["Presence"].sum())
    n_absent  = int((df["Presence"] == 0).sum())
    ratio     = n_absent / n_present if n_present > 0 else 0

    log.info(f"    Presence = 1 (sightings):      {n_present:>7,}")
    log.info(f"    Presence = 0 (pseudo-absences): {n_absent:>7,}")
    log.info(f"    Ratio (0:1):                    {ratio:.2f}:1")

    if 3.5 <= ratio <= 4.5:
        log.info("    ✓ Balance is within expected range (≈4:1)\n")
    else:
        log.warning(f"    ⚠ Unexpected ratio — expected ≈4:1, got {ratio:.2f}:1\n")

    # ── 2. Completeness ──────────────────────────────────────────────────
    log.info("─── 2. DATA COMPLETENESS ─────────────────────────────────────")
    log.info(f"    {'Column':<15} {'Valid':>8} {'Missing':>8} {'Coverage':>10}")
    log.info(f"    {'─'*15} {'─'*8} {'─'*8} {'─'*10}")

    for col in env_cols:
        if col not in df.columns:
            log.info(f"    {col:<15} {'N/A':>8} {'N/A':>8} {'MISSING':>10}")
            continue
        valid   = int(df[col].notna().sum())
        missing = int(df[col].isna().sum())
        pct     = (valid / total) * 100
        status  = "✓" if pct > 90 else ("⚠" if pct > 50 else "✗")
        log.info(f"    {col:<15} {valid:>8,} {missing:>8,} {pct:>8.1f}% {status}")

    log.info("")

    # ── 3. Physical Plausibility ─────────────────────────────────────────
    log.info("─── 3. PHYSICAL PLAUSIBILITY ─────────────────────────────────")

    # Define valid physical ranges for each variable
    plausibility = {
        "SST":         {"unit": "°C",  "min": -2.0,  "max": 35.0,
                        "desc": "Sea Surface Temperature"},
        "Chlorophyll": {"unit": "mg/m³", "min": 0.0, "max": 100.0,
                        "desc": "Chlorophyll-a concentration"},
        "Salinity":    {"unit": "PSU", "min": 20.0,  "max": 40.0,
                        "desc": "Sea surface salinity"},
        "Bathymetry":  {"unit": "m",   "min": -11000, "max": 100,
                        "desc": "Ocean depth (negative = below sea level)"},
    }

    for col, limits in plausibility.items():
        if col not in df.columns or df[col].isna().all():
            log.info(f"    {col} ({limits['desc']})")
            log.info(f"      ✗ No valid data — skipping plausibility check\n")
            continue

        col_min  = df[col].min()
        col_max  = df[col].max()
        col_mean = df[col].mean()

        log.info(f"    {col} ({limits['desc']})")
        log.info(f"      Range: [{col_min:.3f}, {col_max:.3f}] {limits['unit']}")
        log.info(f"      Mean:  {col_mean:.3f} {limits['unit']}")
        log.info(f"      Expected: [{limits['min']}, {limits['max']}] {limits['unit']}")

        in_range = col_min >= limits["min"] and col_max <= limits["max"]
        if in_range:
            log.info(f"      ✓ Values are physically plausible\n")
        else:
            n_outliers = int(
                ((df[col] < limits["min"]) | (df[col] > limits["max"])).sum()
            )
            log.warning(
                f"      ⚠ {n_outliers:,} values outside expected physical range\n"
            )

    log.info("─── SANITY CHECK COMPLETE ────────────────────────────────────\n")


# =========================================================================
#  Part 2 — SST Gradient / Thermal Front Engineering
# =========================================================================

def _compute_gradient_magnitude(sst_slab: xr.DataArray) -> xr.DataArray:
    """
    Compute the spatial gradient magnitude of a 2D SST field in °C/km.

    Uses np.gradient for central differences along both lat/lon axes,
    then converts from °C/pixel to °C/km using the known MUR resolution.

    The conversion accounts for latitude-dependent longitude spacing:
      - 1 pixel in latitude  = 0.01° × 111.32 km/° ≈ 1.1132 km
      - 1 pixel in longitude = 0.01° × 111.32 × cos(lat) km
    """
    sst_values = sst_slab.values

    # Get latitude values for cosine correction
    lats = sst_slab.latitude.values
    lat_mean = np.mean(lats)
    cos_lat = np.cos(np.radians(lat_mean))

    # Pixel spacing in km
    dy_km = PIXEL_SIZE_DEG * 111.32          # latitude direction
    dx_km = PIXEL_SIZE_DEG * 111.32 * cos_lat  # longitude direction

    # Central-difference gradient (°C / km)
    grad_y, grad_x = np.gradient(sst_values, dy_km, dx_km)

    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Return as DataArray with same coordinates
    return xr.DataArray(
        magnitude,
        coords=sst_slab.coords,
        dims=sst_slab.dims,
        name="SST_Gradient",
    )


def _fetch_sst_slab_with_retry(ds, time_key, lat_min, lat_max, lon_min, lon_max):
    """Fetch an SST slab with exponential backoff retries."""
    for attempt in range(MAX_RETRIES):
        try:
            slab = ds[SST_VARIABLE].sel(
                time=time_key, method="nearest"
            ).sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max),
            )
            return slab.load()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep_time = (2 ** attempt) * 5
                log.debug(f"    Fetch failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                log.warning(f"    Failed after {MAX_RETRIES} attempts: {e}")
                return None


def compute_thermal_fronts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 3: For each unique date, fetch the MUR SST slab covering
    all points for that date, compute the spatial gradient, and extract
    the gradient value at each point's coordinates.

    Adds columns: SST_Gradient (°C/km), Is_Thermal_Front (bool)
    """
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║      PHASE 3 — THERMAL FRONT FEATURE ENGINEERING           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    # Initialize output columns
    df["SST_Gradient"]     = np.nan
    df["Is_Thermal_Front"] = False

    # Skip rows where SST is already NaN (no point computing gradient)
    valid_mask = df["SST"].notna()
    n_valid    = int(valid_mask.sum())
    n_skip     = int((~valid_mask).sum())
    log.info(f"  Points with valid SST: {n_valid:,}")
    log.info(f"  Points without SST (will be NaN): {n_skip:,}\n")

    # ── Connect to MUR SST ──────────────────────────────────────────────
    log.info(f"  Connecting to MUR SST: {SST_URL}")
    try:
        ds = xr.open_dataset(SST_URL, engine="netcdf4")
        log.info(f"    ✓ Connected — dims: {dict(ds.dims)}\n")
    except Exception as e:
        log.error(f"    ✗ Failed to connect: {e}")
        log.error("    Cannot compute thermal fronts. Returning dataset as-is.")
        return df

    # ── Group by date and process ────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    temporal_keys = df["Date"].dt.normalize() + pd.Timedelta(hours=9)

    # Only process dates that have valid SST
    df_valid = df[valid_mask].copy()
    df_valid["_time_key"] = temporal_keys[valid_mask]

    unique_keys = sorted(df_valid["_time_key"].unique())
    n_groups = len(unique_keys)
    log.info(f"  Processing {n_groups} unique dates for SST gradient\n")

    t_start = time.time()
    extracted_count = 0

    for g_idx, key in enumerate(unique_keys):
        mask_group = df_valid["_time_key"] == key
        group = df_valid[mask_group]

        lats = group["Lat"].values
        lons = group["Lon"].values

        lat_min = float(lats.min()) - SLAB_BUFFER_DEG
        lat_max = float(lats.max()) + SLAB_BUFFER_DEG
        lon_min = float(lons.min()) - SLAB_BUFFER_DEG
        lon_max = float(lons.max()) + SLAB_BUFFER_DEG

        # Fetch SST slab
        sst_slab = _fetch_sst_slab_with_retry(
            ds, key, lat_min, lat_max, lon_min, lon_max
        )

        if sst_slab is None or sst_slab.size < 4:
            # Need at least a 2×2 grid for gradient
            continue

        # Compute gradient magnitude on the 2D slab
        gradient_field = _compute_gradient_magnitude(sst_slab)

        # Extract gradient values at each point using nearest-neighbor
        coords = {
            "latitude":  xr.DataArray(lats, dims="points"),
            "longitude": xr.DataArray(lons, dims="points"),
        }

        try:
            gradient_values = gradient_field.sel(coords, method="nearest").values
            df.loc[group.index, "SST_Gradient"] = gradient_values
            extracted_count += np.count_nonzero(~np.isnan(gradient_values))
        except Exception as e:
            log.debug(f"    Gradient extraction failed for group {g_idx}: {e}")

        # Progress logging
        if (g_idx + 1) % 50 == 0 or (g_idx + 1) == n_groups:
            elapsed = time.time() - t_start
            pct  = ((g_idx + 1) / n_groups) * 100
            rate = (g_idx + 1) / elapsed if elapsed > 0 else 0
            eta  = (n_groups - g_idx - 1) / rate if rate > 0 else 0
            log.info(
                f"    ↳ Group {g_idx + 1:>5}/{n_groups} "
                f"({pct:5.1f}%) | "
                f"Valid gradients: {extracted_count:,} | "
                f"ETA: {eta:.0f}s"
            )

    ds.close()

    # ── Apply thermal front threshold ────────────────────────────────────
    df["Is_Thermal_Front"] = df["SST_Gradient"] > FRONT_THRESHOLD_C_PER_KM

    elapsed_total = time.time() - t_start
    n_fronts = int(df["Is_Thermal_Front"].sum())
    valid_gradients = int(df["SST_Gradient"].notna().sum())

    log.info(f"\n  ✓ Gradient extraction complete in {elapsed_total:.1f}s")
    log.info(f"    Valid gradients:  {valid_gradients:,}/{len(df):,} "
             f"({(valid_gradients/len(df))*100:.1f}%)")
    log.info(f"    Thermal fronts:   {n_fronts:,} points "
             f"(>{FRONT_THRESHOLD_C_PER_KM} °C/km)")
    log.info(f"    Front prevalence: {(n_fronts/len(df))*100:.1f}% of all points\n")

    # Clean up temp column
    if "_time_key" in df.columns:
        df.drop(columns=["_time_key"], inplace=True)

    return df


# =========================================================================
#  Pipeline Orchestrator
# =========================================================================

def main():
    pipeline_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║   WhaleGuard Phase 3 — Feature Engineering Pipeline        ║")
    log.info("║   Target: North Atlantic Right Whale (Eubalaena glacialis)  ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Load ─────────────────────────────────────────────────────────────
    log.info(f"Loading base dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    # ── Part 1: Sanity Check ─────────────────────────────────────────────
    run_sanity_check(df)

    # ── Part 2: Thermal Front Engineering ────────────────────────────────
    df = compute_thermal_fronts(df)

    # ── Part 3: Export ───────────────────────────────────────────────────
    log.info("─── Exporting Engineered Dataset ──────────────────────────────")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "Date", "Lat", "Lon", "Presence",
        "SST", "Chlorophyll", "Salinity", "Bathymetry",
        "SST_Gradient", "Is_Thermal_Front",
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    df_export = df[output_cols].copy()
    df_export.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - pipeline_start
    log.info(f"\n  Output written to: {OUTPUT_CSV}")
    log.info(f"  Total rows:  {len(df_export):,}")
    log.info(f"  Columns:     {', '.join(output_cols)}")
    log.info("")
    log.info("  Coverage per feature:")
    for col in ["SST", "Chlorophyll", "Salinity", "Bathymetry",
                "SST_Gradient", "Is_Thermal_Front"]:
        if col in df_export.columns:
            valid = int(df_export[col].notna().sum())
            pct   = (valid / len(df_export)) * 100
            log.info(f"    {col:20s}: {valid:>7,}/{len(df_export):,} ({pct:5.1f}%)")

    n_fronts = int(df_export["Is_Thermal_Front"].sum()) if "Is_Thermal_Front" in df_export.columns else 0
    log.info(f"\n  Thermal fronts detected: {n_fronts:,}")
    log.info(f"  Pipeline completed in {elapsed:.1f}s")
    log.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
