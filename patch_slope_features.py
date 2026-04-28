"""
==========================================================================
 WhaleGuard — Phase 5: Spatial Feature Patch
==========================================================================
 Adds three literature-backed spatial features to the existing dataset
 WITHOUT re-running the full ETL pipeline. Uses a single ETOPO1 download
 for all three computations.

 New Features:
   1. Bathy_Slope (m/km)
      Magnitude of the bathymetric depth gradient. Marks the continental
      shelf break and submarine canyons where upwelling aggregates
      copepod prey (Baumgartner & Mate, 2005; Pendleton et al., 2012).

   2. Dist_to_Shore_km
      Haversine distance to the nearest land cell. NARWs stay relatively
      close to shore during calving and migration (Schick et al., 2009).

   3. Dist_to_Shelf_km
      Haversine distance to the nearest 200m isobath (continental shelf
      break). Shelf break upwelling zones aggregate Calanus finmarchicus
      (Roberts et al., 2016).

 Architecture:
   - Downloads ETOPO1 (1-arc-minute, ~1.85 km resolution) ONCE as a
     single static slab covering the full dataset bounding box.
   - Computes all three features from that one grid in memory.
   - Uses scipy.spatial.cKDTree for O(n log n) nearest-neighbor queries.

 Runtime: ~5-10 minutes (mostly the ETOPO download).
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
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", message=".*SerializationWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
#  Configuration
# =========================================================================

INPUT_CSV  = Path("data/processed/ML_Whale_Dataset_Engineered_Patched.csv")
OUTPUT_CSV = Path("data/processed/ML_Whale_Dataset_Final.csv")

# ETOPO1 on NOAA ERDDAP (1-arc-minute global relief)
ETOPO_URL  = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180"
ETOPO_VAR  = "altitude"  # positive = land elevation, negative = ocean depth

# Buffer around bounding box (degrees) to avoid edge effects in gradient
SLAB_BUFFER_DEG = 2.0

# Shelf break defined as the 200m isobath (standard in literature)
SHELF_BREAK_DEPTH = -200  # meters (negative = below sea level)
SHELF_BREAK_TOLERANCE = 50  # meters — cells within ±50m of -200m

# Earth radius for haversine calculations
EARTH_RADIUS_KM = 6371.0

# =========================================================================
#  Logging
# =========================================================================

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("WhaleGuard_Phase5")
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
    fh = logging.FileHandler(log_dir / "phase5_slope.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

log = _setup_logging()


# =========================================================================
#  Helper: Haversine Distance
# =========================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# =========================================================================
#  Step 1: Download ETOPO1 Slab
# =========================================================================

def download_etopo_slab(lat_min, lat_max, lon_min, lon_max):
    """Download a single ETOPO1 slab covering the dataset bounding box."""
    log.info("─── Downloading ETOPO1 Bathymetry Grid ───────────────────────")
    log.info(f"  Source: {ETOPO_URL}")
    log.info(f"  Bounding box: [{lat_min:.1f}, {lat_max:.1f}] × [{lon_min:.1f}, {lon_max:.1f}]")

    t0 = time.time()
    ds = xr.open_dataset(ETOPO_URL)
    slab = ds[ETOPO_VAR].sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    ).load()
    ds.close()

    elapsed = time.time() - t0
    log.info(f"  ✓ Downloaded in {elapsed:.1f}s")
    log.info(f"  Grid shape: {slab.shape} ({slab.shape[0]} lat × {slab.shape[1]} lon)")
    log.info(f"  Depth range: [{float(slab.min()):.0f}, {float(slab.max()):.0f}] m")
    log.info(f"  Resolution: {1/60:.4f}° ≈ 1.85 km\n")

    return slab


# =========================================================================
#  Step 2: Compute Bathymetric Slope
# =========================================================================

def compute_bathy_slope(slab, lats_data, lons_data):
    """
    Compute the bathymetric slope (m/km) at each data point.

    Uses np.gradient on the ETOPO1 grid, converting from m/pixel to m/km
    with latitude-dependent longitude correction.
    """
    log.info("─── Computing Bathymetric Slope ────────────────────────────────")

    depth_values = slab.values.astype(np.float64)
    grid_lats = slab.latitude.values
    grid_lons = slab.longitude.values

    # Pixel spacing in km (ETOPO1 = 1 arc-minute = 1/60°)
    pixel_deg = 1.0 / 60.0
    lat_mean = np.mean(grid_lats)
    cos_lat = np.cos(np.radians(lat_mean))

    dy_km = pixel_deg * 111.32           # latitude direction
    dx_km = pixel_deg * 111.32 * cos_lat # longitude direction

    # Central-difference gradient (m/km)
    grad_y, grad_x = np.gradient(depth_values, dy_km, dx_km)
    slope_field = np.sqrt(grad_x**2 + grad_y**2)

    slope_da = xr.DataArray(
        slope_field,
        coords=slab.coords,
        dims=slab.dims,
        name="Bathy_Slope",
    )

    # Extract at data points using nearest-neighbor
    coords = {
        "latitude":  xr.DataArray(lats_data, dims="points"),
        "longitude": xr.DataArray(lons_data, dims="points"),
    }
    slope_values = slope_da.sel(coords, method="nearest").values

    valid = np.count_nonzero(~np.isnan(slope_values))
    log.info(f"  ✓ Extracted {valid:,}/{len(lats_data):,} slope values")
    log.info(f"  Min: {np.nanmin(slope_values):.3f} m/km")
    log.info(f"  Max: {np.nanmax(slope_values):.3f} m/km")
    log.info(f"  Mean: {np.nanmean(slope_values):.3f} m/km\n")

    return slope_values


# =========================================================================
#  Step 3: Compute Distance to Shore
# =========================================================================

def compute_dist_to_shore(slab, lats_data, lons_data):
    """
    For each data point, find the nearest land cell in the ETOPO1 grid
    and compute haversine distance in km.

    Land = altitude >= 0 in ETOPO1.
    Uses scipy cKDTree for O(n log n) performance.
    """
    log.info("─── Computing Distance to Shore ─────────────────────────────────")

    depth_values = slab.values
    grid_lats = slab.latitude.values
    grid_lons = slab.longitude.values

    # Create meshgrid
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    # Find land cells (altitude >= 0)
    land_mask = depth_values >= 0
    n_land = int(np.sum(land_mask))
    n_ocean = int(np.sum(~land_mask))
    log.info(f"  Grid cells: {depth_values.size:,} total ({n_land:,} land, {n_ocean:,} ocean)")

    if n_land == 0:
        log.warning("  ⚠ No land cells found in slab — all distances will be NaN")
        return np.full(len(lats_data), np.nan)

    # Extract land cell coordinates
    land_lats = lat_mesh[land_mask]
    land_lons = lon_mesh[land_mask]

    # Build KDTree on land cells (using scaled coordinates for approximate distance)
    # Scale longitude by cos(lat) for better distance approximation
    lat_center = np.mean(lats_data)
    cos_center = np.cos(np.radians(lat_center))

    land_coords_scaled = np.column_stack([
        land_lats * 111.32,
        land_lons * 111.32 * cos_center,
    ])

    log.info(f"  Building KDTree with {n_land:,} land cells...")
    tree = cKDTree(land_coords_scaled)

    # Query for each data point
    data_coords_scaled = np.column_stack([
        lats_data * 111.32,
        lons_data * 111.32 * cos_center,
    ])

    log.info(f"  Querying {len(lats_data):,} data points...")
    t0 = time.time()
    _, indices = tree.query(data_coords_scaled, k=1)
    query_time = time.time() - t0
    log.info(f"  ✓ KDTree query completed in {query_time:.1f}s")

    # Compute precise haversine distance to nearest land cell
    nearest_land_lats = land_lats[indices]
    nearest_land_lons = land_lons[indices]
    distances_km = haversine_km(lats_data, lons_data, nearest_land_lats, nearest_land_lons)

    log.info(f"  Min: {np.nanmin(distances_km):.2f} km")
    log.info(f"  Max: {np.nanmax(distances_km):.2f} km")
    log.info(f"  Mean: {np.nanmean(distances_km):.2f} km\n")

    return distances_km


# =========================================================================
#  Step 4: Compute Distance to Shelf Break (200m Isobath)
# =========================================================================

def compute_dist_to_shelf(slab, lats_data, lons_data):
    """
    For each data point, find the nearest cell on the 200m isobath
    (continental shelf break) and compute haversine distance in km.

    Shelf break = cells where depth is between -250m and -150m.
    """
    log.info("─── Computing Distance to Shelf Break (200m Isobath) ─────────")

    depth_values = slab.values
    grid_lats = slab.latitude.values
    grid_lons = slab.longitude.values

    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)

    # Find cells near the 200m isobath
    shelf_mask = (
        (depth_values <= SHELF_BREAK_DEPTH + SHELF_BREAK_TOLERANCE) &
        (depth_values >= SHELF_BREAK_DEPTH - SHELF_BREAK_TOLERANCE)
    )
    n_shelf = int(np.sum(shelf_mask))
    log.info(f"  Shelf break cells (depth {SHELF_BREAK_DEPTH}±{SHELF_BREAK_TOLERANCE}m): {n_shelf:,}")

    if n_shelf == 0:
        log.warning("  ⚠ No shelf break cells found — all distances will be NaN")
        return np.full(len(lats_data), np.nan)

    # Extract shelf break coordinates
    shelf_lats = lat_mesh[shelf_mask]
    shelf_lons = lon_mesh[shelf_mask]

    # Build KDTree
    lat_center = np.mean(lats_data)
    cos_center = np.cos(np.radians(lat_center))

    shelf_coords_scaled = np.column_stack([
        shelf_lats * 111.32,
        shelf_lons * 111.32 * cos_center,
    ])

    log.info(f"  Building KDTree with {n_shelf:,} shelf break cells...")
    tree = cKDTree(shelf_coords_scaled)

    # Query
    data_coords_scaled = np.column_stack([
        lats_data * 111.32,
        lons_data * 111.32 * cos_center,
    ])

    log.info(f"  Querying {len(lats_data):,} data points...")
    t0 = time.time()
    _, indices = tree.query(data_coords_scaled, k=1)
    query_time = time.time() - t0
    log.info(f"  ✓ KDTree query completed in {query_time:.1f}s")

    # Compute precise haversine distance
    nearest_shelf_lats = shelf_lats[indices]
    nearest_shelf_lons = shelf_lons[indices]
    distances_km = haversine_km(lats_data, lons_data, nearest_shelf_lats, nearest_shelf_lons)

    log.info(f"  Min: {np.nanmin(distances_km):.2f} km")
    log.info(f"  Max: {np.nanmax(distances_km):.2f} km")
    log.info(f"  Mean: {np.nanmean(distances_km):.2f} km\n")

    return distances_km


# =========================================================================
#  Main Pipeline
# =========================================================================

def main():
    t_start = time.time()

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  WhaleGuard — Phase 5: Spatial Feature Patch               ║")
    log.info("║  Adding: Bathy_Slope, Dist_to_Shore_km, Dist_to_Shelf_km  ║")
    log.info("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Load Dataset ─────────────────────────────────────────────────────
    log.info("─── Loading Dataset ──────────────────────────────────────────")
    if not INPUT_CSV.exists():
        log.error(f"  ✗ File not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"])
    log.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    log.info(f"  Existing columns: {', '.join(df.columns)}\n")

    lats = df["Lat"].values.astype(np.float64)
    lons = df["Lon"].values.astype(np.float64)

    # ── Download ETOPO1 ──────────────────────────────────────────────────
    lat_min = float(lats.min()) - SLAB_BUFFER_DEG
    lat_max = float(lats.max()) + SLAB_BUFFER_DEG
    lon_min = float(lons.min()) - SLAB_BUFFER_DEG
    lon_max = float(lons.max()) + SLAB_BUFFER_DEG

    slab = download_etopo_slab(lat_min, lat_max, lon_min, lon_max)

    # ── Feature 1: Bathymetric Slope ─────────────────────────────────────
    df["Bathy_Slope"] = compute_bathy_slope(slab, lats, lons)

    # ── Feature 2: Distance to Shore ─────────────────────────────────────
    df["Dist_to_Shore_km"] = compute_dist_to_shore(slab, lats, lons)

    # ── Feature 3: Distance to Shelf Break ───────────────────────────────
    df["Dist_to_Shelf_km"] = compute_dist_to_shelf(slab, lats, lons)

    # ── Export ────────────────────────────────────────────────────────────
    log.info("─── Exporting Final Dataset ──────────────────────────────────")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t_start
    log.info(f"  ✓ Saved to: {OUTPUT_CSV}")
    log.info(f"  Rows: {len(df):,}")
    log.info(f"  Columns ({len(df.columns)}): {', '.join(df.columns)}")
    log.info(f"\n  New feature coverage:")
    for col in ["Bathy_Slope", "Dist_to_Shore_km", "Dist_to_Shelf_km"]:
        valid = int(df[col].notna().sum())
        pct = (valid / len(df)) * 100
        log.info(f"    {col:20s}: {valid:>7,}/{len(df):,} ({pct:5.1f}%)")

    log.info(f"\n  ✓ Patch complete in {elapsed:.1f}s")
    log.info("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
