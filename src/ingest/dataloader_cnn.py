import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import rasterio
import rasterio.crs
from rasterio.enums import Resampling # For resampling methods
from rasterio.warp import calculate_default_transform, reproject
import warnings
import rioxarray # Use rioxarray for easier loading/clipping/resampling
import xarray as xr # Needed for DataArray type hint

# --- Import centralized utils --- #
from .data_utils import (
    determine_target_grid_size, # Keep for potentially calculating UHI grid size
    compute_grid_cell_coordinates,
    precompute_uhi_grids,
    normalize_lst, # Still useful for LST
    get_closest_weather_data, # Weather logic remains
    build_weather_grid, # Weather grid building remains
    normalize_clay_timestamp, # Clay utils remain
    normalize_clay_latlon,
    WEATHER_NORM_PARAMS, # Constant for weather normalization
    resample_xarray_to_target # NEW Resampling utility
)
# ------------------------------ #

# Suppress pandas future warning about timezone parsing
warnings.filterwarnings("ignore", category=FutureWarning, message=".*un-recognized timezone.*")

# ---------------------------------------- #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Band Index Mapping --- #
DEFAULT_MOSAIC_BANDS_ORDER = [
    "blue", "green", "red", "nir", "swir16", "swir22",
]
# -------------------------- #

class CityDataSet(Dataset):
    """
    Dataset for UHI modeling using locally stored data, adapted for CNN model.
    Resamples ALL spatial features (weather, LST, mosaic, DEM, DSM) to a common
    `feature_resolution_m` before returning.
    """
    def __init__(self, bounds: List[float],
                 feature_resolution_m: int, # NEW: Common feature resolution
                 uhi_grid_resolution_m: int, # Resolution of the output UHI grid
                 uhi_csv: str,
                 bronx_weather_csv: str, manhattan_weather_csv: str,
                 data_dir: str, city_name: str,
                 # --- Feature Flags & Paths --- #
                 feature_flags: Dict[str, bool],
                 sentinel_bands_to_load: List[str],
                 dem_path: Optional[str] = None,
                 dsm_path: Optional[str] = None,
                 elevation_nodata: Optional[float] = None,
                 cloudless_mosaic_path: Optional[str] = None,
                 single_lst_median_path: Optional[str] = None,
                 lst_nodata: Optional[float] = None,
                 target_crs_str: str = "EPSG:4326",
                 # --- Other Params --- #
                 ): # Removed low-res/high-res elev path/nodata args
        """
        Initialize the CNN dataset with common feature resampling.

        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]. REQUIRED.
            feature_resolution_m: Target spatial resolution (meters) for ALL input features.
            uhi_grid_resolution_m: Spatial resolution (meters) of the ground truth UHI grid.
            uhi_csv: Path to UHI data CSV.
            bronx_weather_csv: Path to Bronx weather station CSV.
            manhattan_weather_csv: Path to Manhattan weather station CSV.
            data_dir: Base directory for stored data.
            city_name: Name of the city.
            feature_flags (Dict[str, bool]): Controls feature inclusion (use_dem, use_dsm, use_clay, ...).
            sentinel_bands_to_load (List[str]): Bands to load if sentinel_composite used.
            dem_path (Optional[str]): Path to DEM GeoTIFF file.
            dsm_path (Optional[str]): Path to DSM GeoTIFF file.
            elevation_nodata (Optional[float]): Nodata value for DEM/DSM files.
            cloudless_mosaic_path (Optional[str]): Path to mosaic .npy file.
            single_lst_median_path (Optional[str]): Path to LST median .npy file.
            lst_nodata (Optional[float]): Nodata value for LST file.
            target_crs_str (str): Target Coordinate Reference System string (Default: "EPSG:4326").
        """
        # --- Basic Parameters --- #
        assert bounds and len(bounds) == 4, "Bounds must be provided."
        self.bounds = bounds
        self.feature_resolution_m = feature_resolution_m
        self.uhi_grid_resolution_m = uhi_grid_resolution_m
        self.data_dir = Path(data_dir)
        self.city_name = city_name
        self.target_crs_str = target_crs_str
        self.target_crs = rasterio.crs.CRS.from_string(self.target_crs_str)
        self.elevation_nodata = elevation_nodata
        self.lst_nodata = lst_nodata
        # CNN Model doesn't use weather sequences, only target timestamp weather

        self.feature_flags = feature_flags
        self.selected_mosaic_bands = sentinel_bands_to_load
        self._dem_path = dem_path
        self._dsm_path = dsm_path
        self._cloudless_mosaic_path = cloudless_mosaic_path
        self._single_lst_median_path = single_lst_median_path

        # --- Determine Target Grid Size for FEATURES --- #
        self.feat_H, self.feat_W = determine_target_grid_size(self.bounds, self.feature_resolution_m)
        self.feat_transform = rasterio.transform.from_bounds(*self.bounds, self.feat_W, self.feat_H)
        logging.info(f"Target FEATURE grid size (H, W): ({self.feat_H}, {self.feat_W}) @ {self.feature_resolution_m}m, CRS: {self.target_crs_str}")

        # --- Determine Target Grid Size for UHI output --- #
        self.uhi_H, self.uhi_W = determine_target_grid_size(self.bounds, self.uhi_grid_resolution_m)
        logging.info(f"Target UHI grid size (H, W): ({self.uhi_H}, {self.uhi_W}) @ {self.uhi_grid_resolution_m}m")

        # --- Load UHI Data & Precompute UHI Grids/Masks (at UHI resolution) --- #
        self.uhi_data = pd.read_csv(uhi_csv)
        timestamp_col_name = 'datetime'
        if timestamp_col_name not in self.uhi_data.columns:
            raise ValueError(f"Timestamp column ('{timestamp_col_name}') not found in {uhi_csv}.")
        uhi_dt_format = '%d-%m-%Y %H:%M'
        target_timezone = 'US/Eastern'
        try:
            all_timestamps_naive = pd.to_datetime(self.uhi_data[timestamp_col_name], format=uhi_dt_format)
            all_timestamps = all_timestamps_naive.dt.tz_localize(target_timezone, ambiguous='infer')
        except ValueError:
             logging.warning(f"Failed UHI timestamp format {uhi_dt_format}. Trying default parsing...")
             all_timestamps_naive = pd.to_datetime(self.uhi_data[timestamp_col_name], errors='coerce')
             if all_timestamps_naive.isnull().any(): raise ValueError("Failed UHI timestamp parsing.")
             try:
                all_timestamps = all_timestamps_naive.dt.tz_localize(target_timezone, ambiguous='infer')
             except Exception as loc_e: raise ValueError(f"Failed to localize UHI timestamps: {loc_e}")

        self.unique_timestamps = sorted(all_timestamps.unique())
        self.uhi_data[timestamp_col_name] = all_timestamps

        self.target_grids, self.valid_masks = precompute_uhi_grids(
            uhi_data=self.uhi_data,
            bounds=self.bounds,
            sat_H=self.uhi_H, # Use UHI grid size
            sat_W=self.uhi_W,
            resolution_m=self.uhi_grid_resolution_m
        )
        # --- END UHI --- #

        # --- Load Static Data Paths/Objects (will be resampled in __getitem__) --- #

        # 1. DEM (Load with rioxarray, keep as object)
        self.dem_xr = None
        if self.feature_flags.get("use_dem", False):
            if not self._dem_path: raise ValueError("dem_path required if use_dem is True.")
            dem_p = Path(self._dem_path)
            if dem_p.exists():
                logging.info(f"Loading DEM from: {dem_p}")
                try:
                    self.dem_xr = rioxarray.open_rasterio(dem_p, masked=True)
                    if self.elevation_nodata is not None:
                         self.dem_xr = self.dem_xr.where(self.dem_xr != self.elevation_nodata)
                         self.dem_xr.rio.write_nodata(np.nan, encoded=True, inplace=True)
                    if self.dem_xr.rio.crs != self.target_crs:
                       logging.info(f"Reprojecting DEM from {self.dem_xr.rio.crs} to {self.target_crs_str}")
                    logging.info(f"Clipping DEM to bounds: {self.bounds}")
                    min_lon, min_lat, max_lon, max_lat = self.bounds
                    logging.info(f"Opened DEM (lazy load). Native shape (approx): {self.dem_xr.shape}")
                except Exception as e:
                    logging.error(f"Failed to open/process DEM from {dem_p}: {e}")
                    if self.dem_xr: self.dem_xr.close()
                    self.dem_xr = None
            else:
                logging.warning(f"DEM path specified but not found: {dem_p}")

        # 2. DSM (Load with rioxarray, keep as object)
        self.dsm_xr = None
        if self.feature_flags.get("use_dsm", False):
            if not self._dsm_path: raise ValueError("dsm_path required if use_dsm is True.")
            dsm_p = Path(self._dsm_path)
            if dsm_p.exists():
                logging.info(f"Loading DSM from: {dsm_p}")
                try:
                    self.dsm_xr = rioxarray.open_rasterio(dsm_p, masked=True)
                    if self.elevation_nodata is not None:
                         self.dsm_xr = self.dsm_xr.where(self.dsm_xr != self.elevation_nodata)
                         self.dsm_xr.rio.write_nodata(np.nan, encoded=True, inplace=True)
                    if self.dsm_xr.rio.crs != self.target_crs:
                       logging.info(f"Reprojecting DSM from {self.dsm_xr.rio.crs} to {self.target_crs_str}")
                    logging.info(f"Clipping DSM to bounds: {self.bounds}")
                    min_lon, min_lat, max_lon, max_lat = self.bounds
                    logging.info(f"Opened DSM (lazy load). Native shape (approx): {self.dsm_xr.shape}")
                except Exception as e:
                    logging.error(f"Failed to open/process DSM from {dsm_p}: {e}")
                    if self.dsm_xr: self.dsm_xr.close()
                    self.dsm_xr = None
            else:
                logging.warning(f"DSM path specified but not found: {dsm_p}")

        # --- Calculate Global Min/Max for DEM/DSM (if loaded) ---
        self.global_dem_min, self.global_dem_max = None, None
        if self.dem_xr is not None:
            try:
                logging.info("Calculating global DEM min/max...")
                # Compute min/max, ignoring NaNs potentially introduced by nodata handling
                # Ensure computation happens on the actual data, not lazy representation
                self.global_dem_min = float(np.nanmin(self.dem_xr.values))
                self.global_dem_max = float(np.nanmax(self.dem_xr.values))
                logging.info(f"Global DEM Min: {self.global_dem_min}, Max: {self.global_dem_max}")
            except Exception as e:
                logging.error(f"Failed to compute global DEM stats: {e}. Proceeding without global stats.")

        self.global_dsm_min, self.global_dsm_max = None, None
        if self.dsm_xr is not None:
            try:
                logging.info("Calculating global DSM min/max...")
                self.global_dsm_min = float(np.nanmin(self.dsm_xr.values))
                self.global_dsm_max = float(np.nanmax(self.dsm_xr.values))
                logging.info(f"Global DSM Min: {self.global_dsm_min}, Max: {self.global_dsm_max}")
            except Exception as e:
                logging.error(f"Failed to compute global DSM stats: {e}. Proceeding without global stats.")

        # 3. Cloudless Mosaic (Keep as NumPy array for now, will wrap in xr in getitem)
        self.cloudless_mosaic_full_np = None
        self.mosaic_transform = None # Need transform if loading from npy
        self.mosaic_crs = self.target_crs # Assume mosaic is in target CRS
        needs_mosaic = self.feature_flags["use_sentinel_composite"] or \
                       self.feature_flags["use_ndvi"] or \
                       self.feature_flags["use_ndbi"] or \
                       self.feature_flags["use_ndwi"] or \
                       self.feature_flags["use_clay"]

        if needs_mosaic:
            if not self._cloudless_mosaic_path:
                raise ValueError("cloudless_mosaic_path required if using Sentinel composite, indices, or Clay.")
            mosaic_path = Path(self._cloudless_mosaic_path)
            if not mosaic_path.exists(): raise FileNotFoundError(f"Cloudless mosaic file not found: {mosaic_path}")
            logging.info(f"Loading cloudless mosaic from {mosaic_path} with memory mapping")
            self.cloudless_mosaic_full_np = np.load(mosaic_path, mmap_mode='r')
            mosaic_h_orig, mosaic_w_orig = self.cloudless_mosaic_full_np.shape[1:]
            self.mosaic_transform = rasterio.transform.from_bounds(*self.bounds, mosaic_w_orig, mosaic_h_orig)
            logging.info(f"Loaded mosaic shape (native res): {self.cloudless_mosaic_full_np.shape}")

        # 4. LST Median (Load with rioxarray, keep as object)
        self.lst_xr = None
        if self.feature_flags["use_lst"]:
            if not self._single_lst_median_path: raise ValueError("single_lst_median_path required if use_lst is True.")
            lst_path = Path(self._single_lst_median_path)
            if lst_path.exists():
                logging.info(f"Loading LST from: {lst_path}")
                try:
                    self.lst_xr = rioxarray.open_rasterio(lst_path, masked=True)
                    if self.lst_nodata is not None:
                         self.lst_xr = self.lst_xr.where(self.lst_xr != self.lst_nodata)
                         self.lst_xr.rio.write_nodata(np.nan, encoded=True, inplace=True)
                    if self.lst_xr.rio.crs != self.target_crs:
                       logging.info(f"Reprojecting LST from {self.lst_xr.rio.crs} to {self.target_crs_str}")
                    logging.info(f"Opened LST (lazy load). Native shape (approx): {self.lst_xr.shape}")
                except Exception as e:
                    logging.error(f"Failed LST loading/processing from {lst_path}: {e}")
                    if self.lst_xr: self.lst_xr.close()
                    self.lst_xr = None
            else:
                logging.warning(f"LST path specified but not found: {lst_path}")

        # --- Load Weather Station Data --- #
        self.bronx_weather = pd.read_csv(bronx_weather_csv)
        self.manhattan_weather = pd.read_csv(manhattan_weather_csv)
        target_timezone = 'US/Eastern'
        try:
             dt_naive_or_aware_bronx = pd.to_datetime(self.bronx_weather['datetime'], errors='raise')
             if dt_naive_or_aware_bronx.dt.tz is None: self.bronx_weather['datetime'] = dt_naive_or_aware_bronx.dt.tz_localize(target_timezone, ambiguous='infer')
             elif str(dt_naive_or_aware_bronx.dt.tz) != target_timezone: self.bronx_weather['datetime'] = dt_naive_or_aware_bronx.dt.tz_convert(target_timezone)
             else: self.bronx_weather['datetime'] = dt_naive_or_aware_bronx
        except Exception as e: raise ValueError(f"Error processing Bronx weather datetimes: {e}")
        try:
             dt_naive_or_aware_man = pd.to_datetime(self.manhattan_weather['datetime'], errors='raise')
             if dt_naive_or_aware_man.dt.tz is None: self.manhattan_weather['datetime'] = dt_naive_or_aware_man.dt.tz_localize(target_timezone, ambiguous='infer')
             elif str(dt_naive_or_aware_man.dt.tz) != target_timezone: self.manhattan_weather['datetime'] = dt_naive_or_aware_man.dt.tz_convert(target_timezone)
             else: self.manhattan_weather['datetime'] = dt_naive_or_aware_man
        except Exception as e: raise ValueError(f"Error processing Manhattan weather datetimes: {e}")
        self.bronx_coords = (40.872, -73.893)
        self.manhattan_coords = (40.767, -73.964)
        logging.info(f"Loaded Bronx weather data: {len(self.bronx_weather)} records")
        logging.info(f"Loaded Manhattan weather data: {len(self.manhattan_weather)} records")
        # --- END Weather Loading --- #

        # --- Compute Grid Cell Center Coordinates for WEATHER GRID (at FEATURE resolution) --- #
        # These are used for interpolating weather station data to the feature grid
        self.grid_cell_center_lon_feat_res, self.grid_cell_center_lat_feat_res = compute_grid_cell_coordinates(
            self.bounds, self.feat_H, self.feat_W, self.target_crs
        )
        self.weather_grid_coords = np.stack(
            [self.grid_cell_center_lon_feat_res.ravel(), self.grid_cell_center_lat_feat_res.ravel()], axis=-1
        )
        logging.info("Computed grid cell center coordinates for weather grid at feature resolution.")
        # --- END Weather Grid Coords --- #

        # --- Final Log --- #
        logging.info(f"Dataset initialized for {self.city_name} with {len(self)} unique timestamps.")
        logging.info(f"Enabled features (flags): {json.dumps(self.feature_flags)}")
        logging.info(f"DEM loaded: {self.dem_xr is not None}")
        logging.info(f"DSM loaded: {self.dsm_xr is not None}")
        logging.info(f"LST loaded: {self.lst_xr is not None}")
        logging.info(f"Mosaic loaded: {self.cloudless_mosaic_full_np is not None}")

    def __len__(self):
        return len(self.unique_timestamps)

    def __getitem__(self, idx) -> Dict[str, Any]:
        target_timestamp = self.unique_timestamps[idx]

        # --- Retrieve Precomputed UHI Data (at UHI resolution) --- #
        target_grid = self.target_grids[target_timestamp]
        valid_mask = self.valid_masks[target_timestamp]

        # --- Build Weather Grid for target timestamp (at FEATURE resolution) ON THE FLY --- #
        weather_grid_feat_res = build_weather_grid(
            timestamp=target_timestamp,
            bronx_weather=self.bronx_weather,
            manhattan_weather=self.manhattan_weather,
            bronx_coords=self.bronx_coords,
            manhattan_coords=self.manhattan_coords,
            grid_coords=self.weather_grid_coords, # Use feature res coords
            sat_H=self.feat_H, # Use feature res dimensions
            sat_W=self.feat_W
        )
        if weather_grid_feat_res is None:
            # This might happen if weather data is missing for the exact timestamp
            # Handle this case, e.g., by logging and returning a zero grid or raising an error
            logging.error(f"Failed to build weather grid for {target_timestamp}. Weather data might be missing.")
            # Option 1: Raise error
            # raise RuntimeError(f"Failed to build weather grid for {target_timestamp}")
            # Option 2: Return a zero grid (ensure dimensions match)
            weather_grid_feat_res = np.zeros((6, self.feat_H, self.feat_W), dtype=np.float32) # Assuming 6 weather channels

        # --- Prepare Static Features (resampled to FEATURE resolution) --- #
        static_features_list = [] # For non-Clay, non-Elev features
        feature_names = []      # Track names

        # 1. Cloudless Mosaic & Derived Indices
        mosaic_feat_res = None
        needs_mosaic = self.feature_flags["use_sentinel_composite"] or \
                       self.feature_flags["use_ndvi"] or \
                       self.feature_flags["use_ndbi"] or \
                       self.feature_flags["use_ndwi"] or \
                       self.feature_flags["use_clay"]

        if needs_mosaic and self.cloudless_mosaic_full_np is not None:
            coords = {
                'band': DEFAULT_MOSAIC_BANDS_ORDER[:self.cloudless_mosaic_full_np.shape[0]],
                'y': np.linspace(self.bounds[3], self.bounds[1], self.cloudless_mosaic_full_np.shape[1]),
                'x': np.linspace(self.bounds[0], self.bounds[2], self.cloudless_mosaic_full_np.shape[2]),
            }
            mosaic_xr = xr.DataArray(
                self.cloudless_mosaic_full_np,
                coords=coords,
                dims=['band', 'y', 'x'],
                name='mosaic'
            )
            mosaic_xr = mosaic_xr.rio.write_crs(self.mosaic_crs)
            mosaic_xr = mosaic_xr.rio.write_transform(self.mosaic_transform)

            mosaic_feat_res = resample_xarray_to_target(
                mosaic_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs
            )
            if mosaic_feat_res is None:
                 logging.warning(f"Mosaic resampling failed for timestamp {target_timestamp}. Skipping mosaic features.")
                 needs_mosaic = False
            else:
                 if self.feature_flags.get("use_sentinel_composite", False) and self.selected_mosaic_bands:
                     selected_indices = []
                     selected_band_names_ordered = []
                     for band_name in self.selected_mosaic_bands:
                         try:
                             idx = DEFAULT_MOSAIC_BANDS_ORDER.index(band_name)
                             selected_indices.append(idx)
                             selected_band_names_ordered.append(band_name)
                         except ValueError:
                             logging.warning(f"Requested composite band '{band_name}' not found. Skipping.")
                     if selected_indices:
                          mosaic_subset = mosaic_feat_res[selected_indices, :, :]
                          static_features_list.append(mosaic_subset)
                          feature_names.extend([f"sentinel_{b}" for b in selected_band_names_ordered])

                 available_bands_in_resampled = {band: i for i, band in enumerate(DEFAULT_MOSAIC_BANDS_ORDER[:mosaic_feat_res.shape[0]])}
                 def _calculate_index(index_name, band_num_name, band_den_name):
                    num_idx = available_bands_in_resampled.get(band_num_name)
                    den_idx = available_bands_in_resampled.get(band_den_name)
                    if num_idx is None or den_idx is None: return None
                    numerator_band = mosaic_feat_res[num_idx].astype(np.float32)
                    denominator_band = mosaic_feat_res[den_idx].astype(np.float32)
                    denominator_sum = numerator_band + denominator_band
                    index_map = np.full(denominator_sum.shape, 0.0, dtype=np.float32)
                    valid_mask = np.abs(denominator_sum) > 1e-6
                    index_map[valid_mask] = (numerator_band[valid_mask] - denominator_band[valid_mask]) / denominator_sum[valid_mask]
                    return np.clip(index_map, -1.0, 1.0)[np.newaxis, :, :]

                 if self.feature_flags["use_ndvi"]:
                     ndvi_map = _calculate_index("ndvi", "nir", "red")
                     if ndvi_map is not None: static_features_list.append(ndvi_map); feature_names.append("ndvi")
                 if self.feature_flags["use_ndbi"]:
                     ndbi_map = _calculate_index("ndbi", "swir16", "nir")
                     if ndbi_map is not None: static_features_list.append(ndbi_map); feature_names.append("ndbi")
                 if self.feature_flags["use_ndwi"]:
                     ndwi_map = _calculate_index("ndwi", "green", "nir")
                     if ndwi_map is not None: static_features_list.append(ndwi_map); feature_names.append("ndwi")

        # 2. LST Median (Resample to FEATURE resolution)
        lst_feat_res = None
        if self.feature_flags["use_lst"] and self.lst_xr is not None:
            lst_feat_res_raw = resample_xarray_to_target(
                self.lst_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs, fill_value=0.0
            )
            if lst_feat_res_raw is not None:
                 lst_feat_res = normalize_lst(lst_feat_res_raw, self.feat_H, self.feat_W)
                 static_features_list.append(lst_feat_res)
                 feature_names.append("lst")
            else: logging.warning(f"LST resampling failed for timestamp {target_timestamp}.")

        # 3. DEM (Resample to FEATURE resolution)
        dem_feat_res = None
        if self.feature_flags["use_dem"] and self.dem_xr is not None:
            dem_feat_res = resample_xarray_to_target(
                self.dem_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs, fill_value=0.0
            )
            if dem_feat_res is not None:
                # Robust percentile-based normalization
                dem_values_for_norm = dem_feat_res.squeeze()
                p2 = np.nanpercentile(dem_values_for_norm, 2)
                p98 = np.nanpercentile(dem_values_for_norm, 98)
                dem_feat_res = np.clip(dem_feat_res, p2, p98)
                if (p98 - p2) > 1e-6:
                    dem_feat_res = (dem_feat_res - p2) / (p98 - p2)
                else:
                    dem_feat_res = np.full_like(dem_feat_res, 0.5)
                logging.debug(f"DEM normalized using 2nd/98th percentiles ({p2:.2f}, {p98:.2f})")
                
                static_features_list.append(dem_feat_res)
                feature_names.append("dem")
            else: logging.warning(f"DEM resampling failed for timestamp {target_timestamp}.")

        # 4. DSM (Resample to FEATURE resolution)
        dsm_feat_res = None
        if self.feature_flags["use_dsm"] and self.dsm_xr is not None:
            dsm_feat_res = resample_xarray_to_target(
                self.dsm_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs, fill_value=0.0
            )
            if dsm_feat_res is not None:
                # Robust percentile-based normalization
                dsm_values_for_norm = dsm_feat_res.squeeze()
                p2 = np.nanpercentile(dsm_values_for_norm, 2)
                p98 = np.nanpercentile(dsm_values_for_norm, 98)
                dsm_feat_res = np.clip(dsm_feat_res, p2, p98)
                if (p98 - p2) > 1e-6:
                    dsm_feat_res = (dsm_feat_res - p2) / (p98 - p2)
                else:
                    dsm_feat_res = np.full_like(dsm_feat_res, 0.5)
                logging.debug(f"DSM normalized using 2nd/98th percentiles ({p2:.2f}, {p98:.2f})")
                
                static_features_list.append(dsm_feat_res)
                feature_names.append("dsm")
            else: logging.warning(f"DSM resampling failed for timestamp {target_timestamp}.")

        # Combine ALL static features (excluding Clay mosaic input)
        if not static_features_list:
            combined_static_features = np.zeros((0, self.feat_H, self.feat_W), dtype=np.float32)
        else:
            valid_static_features = []
            for i, feat in enumerate(static_features_list):
                if feat is not None and feat.shape[1:] == (self.feat_H, self.feat_W):
                    valid_static_features.append(feat)
                else: logging.warning(f"Static feature '{feature_names[i]}' shape mismatch/None. Skipping.")
            combined_static_features = np.concatenate(valid_static_features, axis=0).astype(np.float32) if valid_static_features else np.zeros((0, self.feat_H, self.feat_W), dtype=np.float32)

        # --- Prepare Clay Inputs (if needed) --- #
        clay_mosaic_input = None # The mosaic resampled to feature res
        norm_latlon_tensor = None
        norm_time_tensor = None
        if self.feature_flags["use_clay"]:
            if mosaic_feat_res is None: logging.warning("Clay enabled, but mosaic failed. Skipping Clay.")
            else:
                clay_input_band_names = ["blue", "green", "red", "nir"]
                clay_input_indices = []
                available_bands_in_resampled = {band: i for i, band in enumerate(DEFAULT_MOSAIC_BANDS_ORDER[:mosaic_feat_res.shape[0]])}
                try:
                    for band_name in clay_input_band_names:
                        clay_input_indices.append(available_bands_in_resampled[band_name])
                    clay_mosaic_input = mosaic_feat_res[clay_input_indices, :, :]
                except KeyError as e: raise ValueError(f"Cannot extract Clay bands ('{e}') from resampled mosaic.")
                
                # --- MODIFIED: Calculate center lat/lon and normalize for Clay ---
                center_lon = (self.bounds[0] + self.bounds[2]) / 2
                center_lat = (self.bounds[1] + self.bounds[3]) / 2
                norm_latlon_tensor = self._normalize_latlon_for_clay_scalar(center_lat, center_lon) # Shape (4,)
                # --- END MODIFIED ---

                norm_time_tensor = normalize_clay_timestamp(target_timestamp) # Shape (4,)

        # --- Assemble Sample Dictionary --- #
        sample = {
            'weather': torch.from_numpy(weather_grid_feat_res).float(), # (C_weather, H_feat, W_feat)
            'target': torch.from_numpy(target_grid).float().unsqueeze(0),   # (1, H_uhi, W_uhi)
            'mask': torch.from_numpy(valid_mask).bool().unsqueeze(0),       # (1, H_uhi, W_uhi)
        }

        # Add combined static features
        if combined_static_features.shape[0] > 0:
            sample['static_features'] = torch.from_numpy(combined_static_features).float()

        # Add optional Clay inputs
        if self.feature_flags["use_clay"] and clay_mosaic_input is not None and \
           norm_latlon_tensor is not None and norm_time_tensor is not None: # Added None checks
            sample['clay_mosaic'] = torch.from_numpy(clay_mosaic_input).float()
            sample['norm_latlon'] = torch.from_numpy(norm_latlon_tensor).float() # Will be (4,)
            sample['norm_timestamp'] = torch.from_numpy(norm_time_tensor).float() # Will be (4,)

        # REMOVE HIGH-RES ELEVATION OUTPUT

        return sample

    def __del__(self):
        if hasattr(self, 'dem_xr') and self.dem_xr is not None:
            try:
                self.dem_xr.close()
            except Exception as e:
                logging.warning(f"Exception closing DEM file handle: {e}")
        if hasattr(self, 'dsm_xr') and self.dsm_xr is not None:
            try:
                self.dsm_xr.close()
            except Exception as e:
                logging.warning(f"Exception closing DSM file handle: {e}")
        if hasattr(self, 'lst_xr') and self.lst_xr is not None:
            try:
                self.lst_xr.close()
            except Exception as e:
                logging.warning(f"Exception closing LST file handle: {e}")

    def _normalize_latlon_for_clay_scalar(self, lat: float, lon: float) -> np.ndarray:
        """Normalizes a single lat/lon pair for Clay model input."""
        lat_rad = lat * np.pi / 180
        lon_rad = lon * np.pi / 180
        return np.array([math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)], dtype=np.float32)