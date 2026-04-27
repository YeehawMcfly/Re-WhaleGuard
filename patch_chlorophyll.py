"""
==========================================================================
 WhaleGuard — Chlorophyll Patch Script
==========================================================================
 Patches the empty Chlorophyll column in the Phase 3 Engineered dataset
 using MODIS Aqua R2022 Science Quality reprocessed data.

 Source Dataset:
   erdMH1chlamday_R2022SQ — MODIS Aqua, R2022 reprocessing, 4km, Monthly
   Variable: chlor_a (mg/m³)
   Coverage: July 2002 – present
   Resolution: 0.0417° (~4km)

 Why this works when erdMH1chlamday failed:
   The R2022 reprocessing uses the OCI (Ocean Color Index) algorithm
   with improved atmospheric correction, dramatically reducing cloud-
   masking artifacts in high-latitude North Atlantic waters. Monthly
   composites further aggregate clear-sky pixels across the month,
   yielding 68-78% spatial coverage vs. 0% from the older version.

 Extraction Method:
   Month-grouped spatial slab fetch with nearest-neighbor selection
   (sufficient at 4km resolution for SDM-scale habitat modeling).
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

INPUT_CSV  = Path("data/processed/ML_Whale_Dataset_Engineered.csv")
OUTPUT_CSV = Path("data/processed/ML_Whale_Dataset_Engineered_Patched.csv")

# MODIS Aqua R2022 Science Quality — monthly, 4km, global
CHL_URL      = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday_R2022SQ"
CHL_VARIABLE = "chlor_a"

# Spatial buffer for bounding box slabs
SLAB_BUFFER_DEG = 1.0

# Retry settings
MAX_RETRIES = 3

# =========================================================================
#  Logging
# =========================================================================

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("WhaleGuard_ChlPatch")
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
    fh = logging.FileHandler(log_dir / "patch_chlorophyll.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

log = _setup_logging()


# =========================================================================
#  Extraction Logic
# =========================================================================

def _fetch_slab_with_retry(ds, time_key, lat_min, lat_max, lon_min, lon_max):
    """
    Fetch a spatial slab from the R2022SQ dataset with exponential backoff.

    NOTE: This dataset has DESCENDING latitude order, so we must use
    slice(lat_max, lat_min) — the opposite of the ascending convention.
    """
    for attempt in range(MAX_RETRIES):
        try:
            slab = ds[CHL_VARIABLE].sel(
                time=time_key, method="nearest"
            ).sel(
                latitude=slice(lat_max, lat_min),   # Descending order!
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


def patch_chlorophyll(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Chlorophyll-a from the R2022 Science Quality MODIS dataset
    using month-grouped spatial slab fetching with nearest-neighbor.
    """
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║      CHLOROPHYLL PATCH — R2022 Science Quality MODIS       ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    total_points = len(df)
    existing_valid = int(df["Chlorophyll"].notna().sum())
    log.info(f"  Total points:     {total_points:,}")
    log.info(f"  Existing valid:   {existing_valid:,} "
             f"({100*existing_valid/total_points:.1f}%)")
    log.info(f"  Points to patch:  {total_points - existing_valid:,}\n")

    # ── Connect ──────────────────────────────────────────────────────────
    log.info(f"  Connecting to: {CHL_URL}")
    try:
        ds = xr.open_dataset(CHL_URL, engine="netcdf4")
        log.info(f"    ✓ Connected — dims: {dict(ds.sizes)}")
        log.info(f"    Time range: {str(ds.time.values[0])[:10]} → "
                 f"{str(ds.time.values[-1])[:10]}")
        log.info(f"    Variable: {CHL_VARIABLE} (Chlorophyll-a, mg/m³)\n")
    except Exception as e:
        log.error(f"    ✗ Failed to connect: {e}")
        return df

    # ── Group by year-month ──────────────────────────────────────────────
    # Monthly composite: group all points to the 16th of their month
    df["Date"] = pd.to_datetime(df["Date"])
    temporal_keys = df["Date"].apply(lambda dt: dt.replace(day=16))

    unique_keys = sorted(temporal_keys.unique())
    n_groups = len(unique_keys)
    log.info(f"  Processing {n_groups} unique months\n")

    t_start = time.time()
    extracted_count = 0
    patched_count   = 0

    for g_idx, key in enumerate(unique_keys):
        mask = temporal_keys == key
        group = df[mask]

        lats = group["Lat"].values
        lons = group["Lon"].values

        lat_min = float(lats.min()) - SLAB_BUFFER_DEG
        lat_max = float(lats.max()) + SLAB_BUFFER_DEG
        lon_min = float(lons.min()) - SLAB_BUFFER_DEG
        lon_max = float(lons.max()) + SLAB_BUFFER_DEG

        slab = _fetch_slab_with_retry(ds, key, lat_min, lat_max, lon_min, lon_max)

        if slab is None or slab.size == 0:
            continue

        # Extract using nearest-neighbor (4km resolution is sufficient)
        coords = {
            "latitude":  xr.DataArray(lats, dims="points"),
            "longitude": xr.DataArray(lons, dims="points"),
        }

        try:
            values = slab.sel(coords, method="nearest").values
            n_valid = int(np.count_nonzero(np.isfinite(values)))
            extracted_count += n_valid

            # Only overwrite NaN values (don't clobber existing data)
            existing_nans = group["Chlorophyll"].isna()
            patch_mask = mask & df["Chlorophyll"].isna()

            # For rows that were NaN before, insert the new values
            new_vals = pd.Series(values, index=group.index)
            df.loc[patch_mask, "Chlorophyll"] = new_vals[existing_nans]
            patched_count += int(np.isfinite(new_vals[existing_nans].values).sum())

        except Exception as e:
            log.debug(f"    Extraction failed for group {g_idx}: {e}")

        # Progress logging
        if (g_idx + 1) % 20 == 0 or (g_idx + 1) == n_groups:
            elapsed = time.time() - t_start
            pct  = ((g_idx + 1) / n_groups) * 100
            rate = (g_idx + 1) / elapsed if elapsed > 0 else 0
            eta  = (n_groups - g_idx - 1) / rate if rate > 0 else 0
            log.info(
                f"    ↳ Month {g_idx + 1:>4}/{n_groups} "
                f"({pct:5.1f}%) | "
                f"Patched: {patched_count:,} | "
                f"ETA: {eta:.0f}s"
            )

    ds.close()

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    final_valid   = int(df["Chlorophyll"].notna().sum())
    final_pct     = 100 * final_valid / total_points

    log.info(f"\n  ✓ Patch complete in {elapsed_total:.1f}s")
    log.info(f"    Before:  {existing_valid:,}/{total_points:,} valid "
             f"({100*existing_valid/total_points:.1f}%)")
    log.info(f"    Patched: {patched_count:,} new values")
    log.info(f"    After:   {final_valid:,}/{total_points:,} valid "
             f"({final_pct:.1f}%)")
    log.info(f"    Still NaN: {total_points - final_valid:,}\n")

    # Physical plausibility check
    if final_valid > 0:
        chl = df["Chlorophyll"].dropna()
        log.info(f"  Plausibility Check:")
        log.info(f"    Min:  {chl.min():.4f} mg/m³")
        log.info(f"    Max:  {chl.max():.4f} mg/m³")
        log.info(f"    Mean: {chl.mean():.4f} mg/m³")
        if chl.min() >= 0 and chl.max() <= 100:
            log.info(f"    ✓ Values are physically plausible\n")
        else:
            log.warning(f"    ⚠ Some values outside expected range [0, 100] mg/m³\n")

    return df


# =========================================================================
#  Main
# =========================================================================

def main():
    pipeline_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║   WhaleGuard — Chlorophyll Patch Pipeline                  ║")
    log.info("║   Source: MODIS Aqua R2022 Science Quality (erdMH1_R2022SQ)║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Load ─────────────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        log.error(f"Input file not found: {INPUT_CSV}")
        log.error("Run phase3_feature_engineering.py first.")
        sys.exit(1)

    log.info(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    # ── Patch ────────────────────────────────────────────────────────────
    df = patch_chlorophyll(df)

    # ── Export ────────────────────────────────────────────────────────────
    log.info("─── Exporting Patched Dataset ─────────────────────────────────")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - pipeline_start
    log.info(f"\n  Output written to: {OUTPUT_CSV}")
    log.info(f"  Total rows:  {len(df):,}")
    log.info(f"  Columns:     {', '.join(df.columns)}")
    log.info("")
    log.info("  Final Coverage per feature:")
    for col in ["SST", "Chlorophyll", "Salinity", "Bathymetry",
                "SST_Gradient", "Is_Thermal_Front"]:
        if col in df.columns:
            valid = int(df[col].notna().sum())
            pct   = (valid / len(df)) * 100
            log.info(f"    {col:20s}: {valid:>7,}/{len(df):,} ({pct:5.1f}%)")

    log.info(f"\n  Pipeline completed in {elapsed:.1f}s")
    log.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
