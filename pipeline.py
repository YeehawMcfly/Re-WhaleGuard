"""
==========================================================================
 WhaleGuard ETL Pipeline — Species Distribution Model Data Preparation
==========================================================================
 Implements two scientific phases:

   Phase 1 — Pseudo-Absence Generation
     Following Gowan & Ortega-Ortiz (2014), generate synthetic background
     points to eliminate observer bias. Each confirmed sighting (Presence=1)
     spawns 4 pseudo-absences (Presence=0) on the same date, scattered
     within a 300 km radius but outside a 15 km exclusion buffer.
     Points are validated to be over ocean (not land).

   Phase 2 — Vectorized Environmental Extraction
     Uses xarray + OPeNDAP to lazily stream remote NetCDF grids from NOAA
     ERDDAP. Spatial extraction maps satellite data directly onto each
     point's spatiotemporal coordinates. No REST API calls needed.

     Datasets:
       • SST          — jplMURSST41          (0.01°, daily, 2002–present)
       • Chlorophyll-a — erdMH1chlamday       (4 km, monthly, 2003–present)
       • Salinity      — erdSoda331oceanmday  (0.5°, monthly, 1980–2015)
       • Bathymetry    — etopo360             (1 arc-min, static)

 References:
   Mosnier et al. (2025) — Feature variable selection
   Gowan & Ortega-Ortiz (2014) — Pseudo-absence methodology
   Ji et al. (2024) — ML architecture benchmarking
==========================================================================
"""

import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe

# ── Suppress xarray/netCDF4 chunking warnings for cleaner logs ──────────
warnings.filterwarnings("ignore", message=".*SerializationWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
#  Configuration
# =========================================================================

# -- Paths ----------------------------------------------------------------
RAW_CSV        = Path("data/raw/23305_RWSAS.csv")
OUTPUT_CSV     = Path("data/processed/ML_Whale_Dataset_Base.csv")

# -- Pseudo-Absence Parameters (Gowan & Ortega-Ortiz, 2014) ---------------
PA_RATIO              = 4        # 4 pseudo-absences per confirmed sighting
BUFFER_INNER_KM       = 15.0     # Exclusion zone around sighting (km)
BUFFER_OUTER_KM       = 300.0    # Maximum scatter radius (km)
MAX_LAND_RETRIES      = 50       # Max attempts to place a point in the ocean
EARTH_RADIUS_KM       = 6371.0   # Mean Earth radius for haversine math

# -- ERDDAP OPeNDAP Endpoints --------------------------------------------
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

DATASETS = {
    "SST": {
        "url":      f"{ERDDAP_BASE}/jplMURSST41",
        "variable": "analysed_sst",
        "has_time":  True,
        "has_depth": False,
        "temporal":  "daily",
        "lon_convention": "pm180",
        "use_nearest": True,           # 0.01° — nearest is fine, interp too slow
    },
    "Chlorophyll": {
        "url":      f"{ERDDAP_BASE}/erdMH1chlamday",
        "variable": "chlorophyll",
        "has_time":  True,
        "has_depth": False,
        "temporal":  "monthly",
        "lon_convention": "pm180",
    },
    "Salinity": {
        "url":      f"{ERDDAP_BASE}/erdSoda331oceanmday_LonPM180",
        "variable": "salt",
        "has_time":  True,
        "has_depth": True,             # Must select surface depth slice
        "temporal":  "monthly",
        "lon_convention": "pm180",
    },
    "Bathymetry": {
        "url":      f"{ERDDAP_BASE}/etopo360",
        "variable": "altitude",
        "has_time":  False,
        "has_depth": False,
        "temporal":  None,
        "lon_convention": "0to360",    # 0 to 360 — requires conversion
        "use_nearest": True,           # 0.017° — nearest is fine
    },
}

# -- Extraction Batch Size ------------------------------------------------
BATCH_SIZE = 500   # Points per OPeNDAP request to avoid server timeouts

# -- Random Seed for Reproducibility -------------------------------------
RANDOM_SEED = 42

# =========================================================================
#  Logging
# =========================================================================

def _setup_logging() -> logging.Logger:
    """Configure a clean, timestamped logger for pipeline progress."""
    logger = logging.getLogger("WhaleGuard_ETL")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (rotated per run)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "pipeline.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

log = _setup_logging()


# =========================================================================
#  PHASE 1 — Data Loading & Pseudo-Absence Generation
# =========================================================================

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """
    Load the Right Whale Sighting Advisory System (RWSAS) dataset.
    Drops rows missing critical fields and tags all rows as Presence=1.
    """
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║             PHASE 1 — PSEUDO-ABSENCE GENERATION            ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    log.info(f"Loading raw sightings from: {csv_path}")
    df = pd.read_csv(csv_path)

    initial_count = len(df)
    log.info(f"  Raw rows loaded: {initial_count:,}")

    # Rename columns for consistency
    col_map = {
        "SIGHTINGDATE": "Date",
        "LAT":          "Lat",
        "LON":          "Lon",
    }
    df.rename(columns=col_map, inplace=True)

    # Drop rows missing critical spatiotemporal fields
    df.dropna(subset=["Date", "Lat", "Lon"], inplace=True)
    dropped = initial_count - len(df)
    if dropped:
        log.warning(f"  Dropped {dropped:,} rows with missing Date/Lat/Lon")

    # Parse dates — the CSV uses "27-Jan-18" format
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)

    # Tag all confirmed sightings
    df["Presence"] = 1

    log.info(f"  Confirmed sightings after cleaning: {len(df):,}")
    log.info(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    return df


def _random_point_in_annulus(
    lat: float, lon: float, r_inner_km: float, r_outer_km: float, rng: np.random.Generator
) -> tuple[float, float]:
    """
    Generate a single random point within a spherical annulus
    (between r_inner and r_outer km) around (lat, lon).

    Uses inverse-CDF sampling for uniform area distribution within the ring,
    then projects onto the sphere using the destination-point formula.
    """
    # Sample distance uniformly by area within the annulus
    u = rng.uniform(0, 1)
    r_km = np.sqrt(u * (r_outer_km**2 - r_inner_km**2) + r_inner_km**2)

    # Random bearing
    theta = rng.uniform(0, 2 * np.pi)

    # Angular distance in radians
    delta = r_km / EARTH_RADIUS_KM

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Destination-point formula (spherical geometry)
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(delta) +
        np.cos(lat_rad) * np.sin(delta) * np.cos(theta)
    )
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(lat_rad),
        np.cos(delta) - np.sin(lat_rad) * np.sin(new_lat_rad),
    )

    return np.degrees(new_lat_rad), np.degrees(new_lon_rad)


def generate_pseudo_absences(df_presence: pd.DataFrame) -> pd.DataFrame:
    """
    For every confirmed sighting, generate PA_RATIO synthetic
    pseudo-absence points (Presence=0) that:

      1. Share the exact same Date (temporal integrity)
      2. Fall within a 300 km radius but outside a 15 km exclusion buffer
      3. Are validated to be over ocean (not on land)

    Returns the combined DataFrame of presences + pseudo-absences.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    log.info(f"Generating {PA_RATIO} pseudo-absences per sighting "
             f"(buffer: {BUFFER_INNER_KM}–{BUFFER_OUTER_KM} km)")

    pa_rows = []
    n_sightings = len(df_presence)
    land_rejects = 0

    for idx, row in df_presence.iterrows():
        generated = 0
        attempts  = 0

        while generated < PA_RATIO:
            new_lat, new_lon = _random_point_in_annulus(
                row["Lat"], row["Lon"],
                BUFFER_INNER_KM, BUFFER_OUTER_KM, rng,
            )

            # Clamp latitude to valid range
            new_lat = np.clip(new_lat, -90.0, 90.0)
            # Wrap longitude to [-180, 180]
            new_lon = ((new_lon + 180) % 360) - 180

            # Ocean validation using global land mask
            if globe.is_ocean(new_lat, new_lon):
                pa_rows.append({
                    "Date":     row["Date"],
                    "Lat":      round(new_lat, 5),
                    "Lon":      round(new_lon, 5),
                    "Presence": 0,
                })
                generated += 1
            else:
                land_rejects += 1

            attempts += 1
            if attempts > MAX_LAND_RETRIES * PA_RATIO:
                # Safeguard: if a coastal point can't produce enough ocean
                # points, fill remainder with NaN coords (will be filtered)
                log.debug(
                    f"  Could not place all PAs for sighting at "
                    f"({row['Lat']:.2f}, {row['Lon']:.2f}) — "
                    f"placed {generated}/{PA_RATIO}"
                )
                break

        # Progress logging every 2000 sightings
        current = len(pa_rows) // PA_RATIO
        if current % 2000 == 0 and current > 0 and generated == PA_RATIO:
            pct = (current / n_sightings) * 100
            log.info(f"  ↳ Progress: {current:,}/{n_sightings:,} sightings processed ({pct:.1f}%)")

    df_pa = pd.DataFrame(pa_rows)

    log.info(f"  Pseudo-absences generated: {len(df_pa):,}")
    log.info(f"  Land-rejected candidates:  {land_rejects:,}")

    # Combine presences and pseudo-absences
    df_master = pd.concat([df_presence, df_pa], ignore_index=True)
    df_master.sort_values(["Date", "Presence"], ascending=[True, False], inplace=True)
    df_master.reset_index(drop=True, inplace=True)

    log.info(f"  Master DataFrame: {len(df_master):,} rows "
             f"({df_master['Presence'].sum():,} presences, "
             f"{(df_master['Presence'] == 0).sum():,} absences)")

    return df_master


# =========================================================================
#  PHASE 2 — Vectorized Environmental Extraction via OPeNDAP
# =========================================================================
#
#  All temporal datasets are processed by grouping points by their
#  temporal key (date for daily SST, year-month for monthly Chl/Salinity).
#  For each group we:
#    1. Select the nearest time slice from the remote dataset
#    2. Fetch a spatial bounding-box slab covering all group points
#    3. Apply nearest (high-res) or bilinear (coarse-res) extraction
#
#  Static datasets (Bathymetry) are processed in spatial batches.
# =========================================================================

SLAB_BUFFER_DEG = 1.0


def _open_dataset(name: str, config: dict) -> xr.Dataset:
    """Lazily open a remote ERDDAP dataset via OPeNDAP."""
    log.info(f"  Connecting to {name}: {config['url']}")
    try:
        ds = xr.open_dataset(config["url"], engine="netcdf4")
        log.info(f"    ✓ Connected — dims: {dict(ds.dims)}")
        return ds
    except Exception as e:
        log.error(f"    ✗ Failed to connect to {name}: {e}")
        raise


def _extract_slab_group(
    ds: xr.Dataset,
    config: dict,
    df_group: pd.DataFrame,
    time_key,
    max_retries: int = 3,
) -> np.ndarray:
    """
    For a group of points sharing a temporal key:
      1. Select the nearest time slice
      2. Fetch a spatial slab covering the bounding box
      3. Extract values via nearest or bilinear interpolation
      * Includes exponential backoff retries for network resilience
    """
    variable = config["variable"]
    use_nearest = config.get("use_nearest", False)

    lats = df_group["Lat"].values
    lons_raw = df_group["Lon"].values.copy()
    lons = lons_raw % 360 if config["lon_convention"] == "0to360" else lons_raw

    lat_min = float(lats.min()) - SLAB_BUFFER_DEG
    lat_max = float(lats.max()) + SLAB_BUFFER_DEG
    lon_min = float(lons.min()) - SLAB_BUFFER_DEG
    lon_max = float(lons.max()) + SLAB_BUFFER_DEG

    da = ds[variable].isel(depth=0) if config["has_depth"] else ds[variable]

    slab_loaded = None
    for attempt in range(max_retries):
        try:
            if config["has_time"] and time_key is not None:
                slab = da.sel(time=time_key, method="nearest").sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max),
                )
            else:
                slab = da.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max),
                )
            slab_loaded = slab.load()
            break  # Success
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) * 5
                log.debug(f"      Fetch failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                log.warning(f"      Fetch failed after {max_retries} attempts.")
                return np.full(len(df_group), np.nan)

    if slab_loaded is None:
        return np.full(len(df_group), np.nan)

    coords = {
        "latitude":  xr.DataArray(lats, dims="points"),
        "longitude": xr.DataArray(lons, dims="points"),
    }

    if use_nearest:
        try:
            return slab_loaded.sel(coords, method="nearest").values
        except Exception:
            return np.full(len(df_group), np.nan)
    else:
        try:
            return slab_loaded.interp(coords, method="linear").values
        except Exception:
            try:
                return slab_loaded.sel(coords, method="nearest").values
            except Exception:
                return np.full(len(df_group), np.nan)


def _compute_temporal_key(dates: pd.Series, temporal: str):
    """Compute temporal grouping key for date-based slab fetching."""
    if temporal == "daily":
        return dates.dt.normalize() + pd.Timedelta(hours=9)
    elif temporal == "monthly":
        return dates.apply(lambda dt: dt.replace(day=16))
    return None


def extract_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Extract environmental data for all points using OPeNDAP.
    Groups by date/month for temporal datasets, batches for static.
    """
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║          PHASE 2 — ENVIRONMENTAL DATA EXTRACTION           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    total_points = len(df)
    log.info(f"Extracting features for {total_points:,} points across 4 datasets\n")

    for name in DATASETS:
        df[name] = np.nan

    log.info("─── Establishing OPeNDAP Connections ─────────────────────────")
    open_datasets = {}
    for name, config in DATASETS.items():
        try:
            open_datasets[name] = _open_dataset(name, config)
        except Exception:
            log.error(f"  Skipping {name} — will be NaN in output")
    log.info("")

    for name, config in DATASETS.items():
        if name not in open_datasets:
            continue

        ds = open_datasets[name]
        temporal = config.get("temporal")
        use_nearest = config.get("use_nearest", False)
        method_label = "nearest" if use_nearest else "bilinear"
        log.info(f"─── Extracting: {name} ({config['variable']}) ───────────")

        t_start = time.time()
        extracted_count = 0

        if temporal is not None:
            temporal_keys = _compute_temporal_key(df["Date"], temporal)
            unique_keys = sorted(temporal_keys.unique())
            n_groups = len(unique_keys)
            log.info(f"    Strategy: {n_groups} {temporal} slabs × "
                     f"{method_label}")

            for g_idx, key in enumerate(unique_keys):
                mask = temporal_keys == key
                df_group = df[mask]

                values = _extract_slab_group(ds, config, df_group, key)
                df.loc[df_group.index, name] = values
                extracted_count += np.count_nonzero(~np.isnan(values))

                if (g_idx + 1) % 50 == 0 or (g_idx + 1) == n_groups:
                    elapsed = time.time() - t_start
                    pct = ((g_idx + 1) / n_groups) * 100
                    rate = (g_idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_groups - g_idx - 1) / rate if rate > 0 else 0
                    log.info(
                        f"    ↳ Group {g_idx + 1:>5}/{n_groups} "
                        f"({pct:5.1f}%) | "
                        f"Valid: {extracted_count:,} | "
                        f"ETA: {eta:.0f}s"
                    )
        else:
            n_batches = int(np.ceil(total_points / BATCH_SIZE))
            log.info(f"    Strategy: {n_batches} batches × {method_label}")

            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end   = min(start + BATCH_SIZE, total_points)
                df_batch = df.iloc[start:end]

                values = _extract_slab_group(
                    ds, config, df_batch, time_key=None
                )
                df.loc[df_batch.index, name] = values
                extracted_count += np.count_nonzero(~np.isnan(values))

                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
                    elapsed = time.time() - t_start
                    pct = ((batch_idx + 1) / n_batches) * 100
                    rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
                    log.info(
                        f"    ↳ Batch {batch_idx + 1:>4}/{n_batches} "
                        f"({pct:5.1f}%) | "
                        f"Valid: {extracted_count:,} | "
                        f"ETA: {eta:.0f}s"
                    )

        elapsed_total = time.time() - t_start
        valid_pct = (extracted_count / total_points) * 100
        log.info(
            f"    ✓ {name} complete — {extracted_count:,}/{total_points:,} "
            f"valid ({valid_pct:.1f}%) in {elapsed_total:.1f}s\n"
        )

        # ── Checkpointing ────────────────────────────────────────────────
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file = OUTPUT_CSV.with_name(f"checkpoint_{name}.csv")
        df.to_csv(checkpoint_file, index=False)
        log.info(f"    ✓ Checkpoint saved: {checkpoint_file}\n")

        ds.close()

    return df


# =========================================================================
#  Pipeline Orchestrator
# =========================================================================

def main():
    """Execute the full ETL pipeline."""
    pipeline_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║   WhaleGuard ETL Pipeline — SDM Data Preparation           ║")
    log.info("║   Target: North Atlantic Right Whale (Eubalaena glacialis)  ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Phase 1: Load + Pseudo-Absences ──────────────────────────────────
    df_sightings = load_and_clean(RAW_CSV)
    df_master    = generate_pseudo_absences(df_sightings)

    log.info("")

    # ── Phase 2: Environmental Extraction ────────────────────────────────
    df_final = extract_environmental_features(df_master)

    # ── Export ────────────────────────────────────────────────────────────
    log.info("─── Exporting Final Dataset ──────────────────────────────────")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Select and order output columns
    output_cols = [
        "Date", "Lat", "Lon", "Presence",
        "SST", "Chlorophyll", "Salinity", "Bathymetry",
    ]
    # Keep only columns that exist (in case a dataset was skipped)
    output_cols = [c for c in output_cols if c in df_final.columns]

    df_export = df_final[output_cols].copy()
    df_export.to_csv(OUTPUT_CSV, index=False)

    # ── Summary Statistics ───────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    log.info(f"\n  Output written to: {OUTPUT_CSV}")
    log.info(f"  Total rows:        {len(df_export):,}")
    log.info(f"  Columns:           {', '.join(output_cols)}")
    log.info("")
    log.info("  Coverage per feature:")
    for col in ["SST", "Chlorophyll", "Salinity", "Bathymetry"]:
        if col in df_export.columns:
            valid   = df_export[col].notna().sum()
            total   = len(df_export)
            pct     = (valid / total) * 100
            log.info(f"    {col:15s}: {valid:>7,}/{total:,} ({pct:5.1f}%)")

    log.info(f"\n  Pipeline completed in {elapsed:.1f}s")
    log.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
