import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import zoom
import json
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import warnings
import rioxarray

# --- Import centralized utils --- #
from .data_utils import (
    determine_target_grid_size,
    compute_grid_cell_coordinates,
    precompute_uhi_grids,
    load_process_elevation,
    normalize_lst,
    get_closest_weather_data,
    build_weather_grid,
    normalize_clay_timestamp,
    normalize_clay_latlon,
    WEATHER_NORM_PARAMS # Constant for weather normalization
)
# ------------------------------ #

# Suppress pandas future warning about timezone parsing 
warnings.filterwarnings("ignore", category=FutureWarning, message=".*un-recognized timezone.*")

# ---------------------------------------- #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Band Index Mapping --- #
# Assumes input mosaic has these bands in this order if ALL are loaded
# Modify this if your mosaic order is different
DEFAULT_MOSAIC_BANDS_ORDER = [
    "blue", "green", "red", "nir", "swir16", "swir22",
    # Add others like "rededge1", "rededge2", "rededge3" if present and needed
]

# -------------------------- #

class CityDataSetBranched(Dataset):
    """
    Dataset for UHI modeling using locally stored data, adapted for branched model.
    Uses centralized data utility functions for processing.

    Loads satellite features (mosaic subset, derived indices), elevation (DEM/DSM),
    LST, and weather data based on configuration flags.

    Features:
    - Selectable Sentinel-2 bands for the mosaic.
    - Optional calculation and inclusion of NDVI, NDBI, NDWI.
    - Optional inclusion of HIGH-RESOLUTION DEM, DSM (loaded directly).
    - Optional inclusion of LOW-RESOLUTION DEM, DSM (resampled, added to static_features).
    - Optional inclusion of LST (added to static_features).
    - Weather data is always included.
    - Returns static features (low-res non-elev) combined into a single tensor.
    - Returns high-res DEM/DSM as separate tensors.
    """

    def __init__(self, bounds: List[float],
                 resolution_m: int,
                 uhi_csv: str,
                 bronx_weather_csv: str, manhattan_weather_csv: str,
                 data_dir: str, city_name: str,
                 # --- Feature Flags & Paths (from config) ---
                 feature_flags: Dict[str, bool],
                 sentinel_bands_to_load: List[str],
                 # --- LOW-RES Elevation Paths (Optional) --- #
                 dem_path_low_res: Optional[str] = None, # Dir or file for low-res
                 dsm_path_low_res: Optional[str] = None, # Dir or file for low-res
                 # --- HIGH-RES Elevation Paths (Optional) --- #
                 dem_path_high_res: Optional[str] = None, # File path for high-res
                 dsm_path_high_res: Optional[str] = None, # File path for high-res
                 high_res_nodata: Optional[float] = None, # Nodata for high-res files
                 # ------------------------------------------ #
                 cloudless_mosaic_path: Optional[str] = None,
                 single_lst_median_path: Optional[str] = None,
                 # --- UPDATED DEFAULT: Weather Sequence Length ---
                 weather_seq_length: int = 60, # Default set to 60
                 # --- Other Params --- #
                 low_res_elevation_nodata: Optional[float] = None, # Nodata for low-res tiles/files.
                 ):
        """
        Initialize the branched dataset, supporting weather sequences and high-res elevation.

        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]. REQUIRED.
            resolution_m: Target spatial resolution (meters) for LOW-RES features.
            uhi_csv: Path to UHI data CSV.
            bronx_weather_csv: Path to Bronx weather station CSV.
            manhattan_weather_csv: Path to Manhattan weather station CSV.
            data_dir: Base directory for stored data.
            city_name: Name of the city.
            feature_flags (Dict[str, bool]): Dictionary controlling feature inclusion.
                Expected keys: use_dem_high_res, use_dsm_high_res, use_dem_low_res, use_dsm_low_res,
                               use_clay, use_sentinel_composite, use_lst, use_ndvi, use_ndbi, use_ndwi
            sentinel_bands_to_load (List[str]): Bands to load if sentinel_composite used.
            dem_path_low_res (Optional[str]): Path to LOW-RES DEM dir or file.
            dsm_path_low_res (Optional[str]): Path to LOW-RES DSM dir or file.
            dem_path_high_res (Optional[str]): Path to HIGH-RES DEM GeoTIFF file.
            dsm_path_high_res (Optional[str]): Path to HIGH-RES DSM GeoTIFF file.
            high_res_nodata (Optional[float]): Nodata value for high-res files (e.g., np.nan).
            cloudless_mosaic_path (Optional[str]): Path to mosaic .npy file.
            single_lst_median_path (Optional[str]): Path to LST median .npy file.
            weather_seq_length (int): Number of weather time steps (Default: 60).
            low_res_elevation_nodata: Nodata value used in LOW-RES DEM/DSM tiles/files.
        """
        # --- Basic Parameters ---
        assert bounds and len(bounds) == 4, "Bounds [min_lon, min_lat, max_lon, max_lat] must be provided."
        self.bounds = bounds
        self.resolution_m = resolution_m
        self.data_dir = Path(data_dir)
        self.city_name = city_name
        # Determine target CRS implicitly from bounds (assuming WGS84 / EPSG:4326 for lat/lon bounds)
        # If bounds were in a different CRS, this might need adjustment, but EPSG:4326 is standard.
        self.target_crs_str = "EPSG:4326"
        self.target_crs = rasterio.crs.CRS.from_string(self.target_crs_str)
        self.low_res_elevation_nodata = low_res_elevation_nodata
        self.high_res_nodata = high_res_nodata
        self.weather_seq_length = weather_seq_length

        # Store feature flags and related paths from the dictionary
        self.feature_flags = feature_flags
        self.selected_mosaic_bands = sentinel_bands_to_load
        self._dem_path_low_res = dem_path_low_res
        self._dsm_path_low_res = dsm_path_low_res
        self._dem_path_high_res = dem_path_high_res
        self._dsm_path_high_res = dsm_path_high_res
        self._cloudless_mosaic_path = cloudless_mosaic_path
        self._single_lst_median_path = single_lst_median_path

        # --- Load UHI Data & Determine Target Grid Size --- #
        self.uhi_data = pd.read_csv(uhi_csv)
        timestamp_col_name = 'datetime'
        if timestamp_col_name not in self.uhi_data.columns:
            raise ValueError(f"Timestamp column ('{timestamp_col_name}') not found in {uhi_csv}. Found columns: {self.uhi_data.columns.tolist()}")
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
        self.sat_H, self.sat_W = determine_target_grid_size(self.bounds, self.resolution_m)
        self.target_transform = rasterio.transform.from_bounds(*self.bounds, self.sat_W, self.sat_H)
        # --- END UHI & Grid Size --- #

        # --- Feature Loading & Processing --- #
        self.static_features_list = []
        self.feature_names = []

        # --- NEW: High-Resolution Elevation Loading (No Resampling) --- #
        self.high_res_dem_full = None
        if self.feature_flags.get("use_dem_high_res", False):
            if not self._dem_path_high_res: raise ValueError("dem_path_high_res required if use_dem_high_res is True.")
            dem_p = Path(self._dem_path_high_res)
            if dem_p.exists():
                logging.info(f"Loading HIGH-RESOLUTION DEM from: {dem_p}")
                try:
                    rds = rioxarray.open_rasterio(dem_p)
                    if self.high_res_nodata is not None:
                        rds = rds.where(rds != self.high_res_nodata)
                        rds.rio.write_nodata(np.nan, encoded=True, inplace=True)

                    if rds.rio.crs != self.target_crs:
                       logging.info(f"Reprojecting HIGH-RES DEM from {rds.rio.crs} to {self.target_crs_str}")
                       rds = rds.rio.reproject(self.target_crs_str)

                    logging.info(f"Clipping HIGH-RES DEM to bounds: {self.bounds}")
                    min_lon, min_lat, max_lon, max_lat = self.bounds
                    clipped_rds = rds.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)

                    self.high_res_dem_full = clipped_rds.astype(np.float32).to_numpy()
                    if self.high_res_dem_full.ndim == 2:
                        self.high_res_dem_full = self.high_res_dem_full[np.newaxis, :, :]
                    elif self.high_res_dem_full.ndim == 3 and self.high_res_dem_full.shape[0] != 1:
                         logging.warning(f"Loaded high-res DEM has unexpected band dim {self.high_res_dem_full.shape[0]}, taking first band.")
                         self.high_res_dem_full = self.high_res_dem_full[[0], :, :]

                    self.high_res_dem_full = np.nan_to_num(self.high_res_dem_full, nan=0.0) # Fill any remaining NaNs
                    logging.info(f"Loaded HIGH-RES DEM shape: {self.high_res_dem_full.shape}")
                    rds.close(); del rds; del clipped_rds
                except Exception as e:
                    logging.error(f"Failed to load/process high-res DEM from {dem_p}: {e}")
                    self.high_res_dem_full = None # Ensure it's None if loading fails
                    self.feature_flags["use_dem_high_res"] = False # Disable flag if failed
            else:
                logging.warning(f"High-resolution DEM path specified but not found: {dem_p}")
                self.feature_flags["use_dem_high_res"] = False

        self.high_res_dsm_full = None
        if self.feature_flags.get("use_dsm_high_res", False):
            if not self._dsm_path_high_res: raise ValueError("dsm_path_high_res required if use_dsm_high_res is True.")
            dsm_p = Path(self._dsm_path_high_res)
            if dsm_p.exists():
                logging.info(f"Loading HIGH-RESOLUTION DSM from: {dsm_p}")
                try:
                    rds = rioxarray.open_rasterio(dsm_p)
                    if self.high_res_nodata is not None:
                        rds = rds.where(rds != self.high_res_nodata)
                        rds.rio.write_nodata(np.nan, encoded=True, inplace=True)

                    if rds.rio.crs != self.target_crs:
                       logging.info(f"Reprojecting HIGH-RES DSM from {rds.rio.crs} to {self.target_crs_str}")
                       rds = rds.rio.reproject(self.target_crs_str)

                    logging.info(f"Clipping HIGH-RES DSM to bounds: {self.bounds}")
                    min_lon, min_lat, max_lon, max_lat = self.bounds
                    clipped_rds = rds.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)

                    self.high_res_dsm_full = clipped_rds.astype(np.float32).to_numpy()
                    if self.high_res_dsm_full.ndim == 2:
                        self.high_res_dsm_full = self.high_res_dsm_full[np.newaxis, :, :]
                    elif self.high_res_dsm_full.ndim == 3 and self.high_res_dsm_full.shape[0] != 1:
                         logging.warning(f"Loaded high-res DSM has unexpected band dim {self.high_res_dsm_full.shape[0]}, taking first band.")
                         self.high_res_dsm_full = self.high_res_dsm_full[[0], :, :]

                    self.high_res_dsm_full = np.nan_to_num(self.high_res_dsm_full, nan=0.0) # Fill any remaining NaNs
                    logging.info(f"Loaded HIGH-RES DSM shape: {self.high_res_dsm_full.shape}")
                    rds.close(); del rds; del clipped_rds
                except Exception as e:
                    logging.error(f"Failed to load/process high-res DSM from {dsm_p}: {e}")
                    self.high_res_dsm_full = None # Ensure it's None if loading fails
                    self.feature_flags["use_dsm_high_res"] = False # Disable flag if failed
            else:
                logging.warning(f"High-resolution DSM path specified but not found: {dsm_p}")
                self.feature_flags["use_dsm_high_res"] = False
        # --- END High-Res Elevation Loading --- #

        # 1. Cloudless Mosaic & Derived Indices (Based on Flags)
        self.cloudless_mosaic_full = None
        # Need mosaic if using composite OR any index OR clay
        needs_mosaic = self.feature_flags["use_sentinel_composite"] or \
                       self.feature_flags["use_ndvi"] or \
                       self.feature_flags["use_ndbi"] or \
                       self.feature_flags["use_ndwi"] or \
                       self.feature_flags["use_clay"]

        if needs_mosaic:
            if not self._cloudless_mosaic_path:
                raise ValueError("cloudless_mosaic_path is required if using Sentinel composite, indices, or Clay features.")
            mosaic_path = Path(self._cloudless_mosaic_path)
            if not mosaic_path.exists(): raise FileNotFoundError(f"Cloudless mosaic file not found: {mosaic_path}")
            logging.info(f"Loading cloudless mosaic from {mosaic_path}")
            self.cloudless_mosaic_full = np.load(mosaic_path)
            if self.cloudless_mosaic_full.shape[1:] != (self.sat_H, self.sat_W):
                 zoom_factors = (1, self.sat_H / self.cloudless_mosaic_full.shape[1], self.sat_W / self.cloudless_mosaic_full.shape[2])
                 logging.info(f"Resizing full mosaic from {self.cloudless_mosaic_full.shape[1:]} to {(self.sat_H, self.sat_W)}")
                 self.cloudless_mosaic_full = zoom(self.cloudless_mosaic_full, zoom=zoom_factors, order=1)

            # Add selected bands for composite feature
            if self.feature_flags["use_sentinel_composite"]:
                 selected_indices = []
                 selected_band_names_ordered = []
                 for band_name in self.selected_mosaic_bands:
                     try:
                         idx = DEFAULT_MOSAIC_BANDS_ORDER.index(band_name)
                         selected_indices.append(idx)
                         selected_band_names_ordered.append(band_name)
                     except ValueError:
                         logging.warning(f"Requested composite band '{band_name}' not found in default order. Skipping.")
                 if selected_indices:
                     mosaic_subset = self.cloudless_mosaic_full[selected_indices, :, :]
                     self.static_features_list.append(mosaic_subset)
                     self.feature_names.extend([f"sentinel_{b}" for b in selected_band_names_ordered])
                     logging.info(f"Selected {mosaic_subset.shape[0]} Sentinel composite bands: {selected_band_names_ordered}")
                 else:
                     logging.warning("No valid Sentinel composite bands selected. Feature disabled.")

            # Calculate spectral indices if flagged
            available_bands_in_full_mosaic = {band: i for i, band in enumerate(DEFAULT_MOSAIC_BANDS_ORDER)}
            def _calculate_index(index_name, band_num_name, band_den_name):
                 num_idx = available_bands_in_full_mosaic.get(band_num_name)
                 den_idx = available_bands_in_full_mosaic.get(band_den_name)
                 if num_idx is None or den_idx is None:
                     logging.warning(f"Cannot calculate {index_name.upper()}: Required bands ('{band_num_name}', '{band_den_name}') not available.")
                     return None
                 numerator_band = self.cloudless_mosaic_full[num_idx].astype(np.float32)
                 denominator_band = self.cloudless_mosaic_full[den_idx].astype(np.float32)
                 denominator_sum = numerator_band + denominator_band
                 index_map = np.full(denominator_sum.shape, 0.0, dtype=np.float32)
                 valid_mask = np.abs(denominator_sum) > 1e-6
                 index_map[valid_mask] = (numerator_band[valid_mask] - denominator_band[valid_mask]) / denominator_sum[valid_mask]
                 index_map = np.clip(index_map, -1.0, 1.0)
                 logging.info(f"Calculated {index_name.upper()}")
                 return index_map[np.newaxis, :, :]

            if self.feature_flags["use_ndvi"]:
                ndvi_map = _calculate_index("ndvi", "nir", "red")
                if ndvi_map is not None: self.static_features_list.append(ndvi_map); self.feature_names.append("ndvi")
            if self.feature_flags["use_ndbi"]:
                ndbi_map = _calculate_index("ndbi", "swir16", "nir") # Using Sentinel SWIR16
                if ndbi_map is not None: self.static_features_list.append(ndbi_map); self.feature_names.append("ndbi")
            if self.feature_flags["use_ndwi"]:
                 ndwi_map = _calculate_index("ndwi", "green", "nir") # McFeeters version
                 if ndwi_map is not None: self.static_features_list.append(ndwi_map); self.feature_names.append("ndwi")

        # 2. LST Median (Based on Flag)
        if self.feature_flags["use_lst"]:
            if not self._single_lst_median_path: raise ValueError("single_lst_median_path required if use_lst is True.")
            lst_path = Path(self._single_lst_median_path)
            if not lst_path.exists(): raise FileNotFoundError(f"LST median path not found: {lst_path}")
            logging.info(f"Loading LST median from: {lst_path}")
            try:
                lst_tensor = np.load(lst_path)
                if lst_tensor.ndim == 2: lst_tensor = lst_tensor[np.newaxis, :, :]
                if lst_tensor.shape[1:] != (self.sat_H, self.sat_W):
                    zoom_factors = (1, self.sat_H / lst_tensor.shape[1], self.sat_W / lst_tensor.shape[2])
                    logging.info(f"Resizing LST from {lst_tensor.shape[1:]} to {(self.sat_H, self.sat_W)}")
                    lst_tensor = zoom(lst_tensor, zoom=zoom_factors, order=1)
                normalized_lst = normalize_lst(lst_tensor, self.sat_H, self.sat_W)
                self.static_features_list.append(normalized_lst)
                self.feature_names.append("lst")
                logging.info(f"Loaded LST median shape {normalized_lst.shape}")
            except Exception as e:
                logging.error(f"Failed LST loading/processing from {lst_path}: {e}. Disabling LST.")
                self.feature_flags["use_lst"] = False # Update flag if failed

        # 3. LOW-RES DEM (Based on Flag, Added to static_features)
        if self.feature_flags.get("use_dem_low_res", False):
            if not self._dem_path_low_res: raise ValueError("dem_path_low_res required if use_dem_low_res is True.")
            dem_grid = load_process_elevation(self._dem_path_low_res, "DEM", self.bounds, self.resolution_m, self.sat_H, self.sat_W, self.target_crs_str, self.target_transform, self.low_res_elevation_nodata)
            if dem_grid is not None: self.static_features_list.append(dem_grid); self.feature_names.append("dem_low_res")
            else: logging.warning("LOW-RES DEM loading failed. Feature disabled."); self.feature_flags["use_dem_low_res"] = False

        # 4. LOW-RES DSM (Based on Flag, Added to static_features)
        if self.feature_flags.get("use_dsm_low_res", False):
            if not self._dsm_path_low_res: raise ValueError("dsm_path_low_res required if use_dsm_low_res is True.")
            dsm_grid = load_process_elevation(self._dsm_path_low_res, "DSM", self.bounds, self.resolution_m, self.sat_H, self.sat_W, self.target_crs_str, self.target_transform, self.low_res_elevation_nodata)
            if dsm_grid is not None: self.static_features_list.append(dsm_grid); self.feature_names.append("dsm_low_res")
            else: logging.warning("LOW-RES DSM loading failed. Feature disabled."); self.feature_flags["use_dsm_low_res"] = False

        # --- Combine Static Features (Non-Clay) --- #
        if not self.static_features_list:
            logging.warning("No non-Clay static features were loaded or enabled.")
            self.combined_static_features = np.zeros((0, self.sat_H, self.sat_W), dtype=np.float32)
        else:
            for i, feat in enumerate(self.static_features_list):
                if feat.shape[1:] != (self.sat_H, self.sat_W): raise ValueError(f"Static feat '{self.feature_names[i]}' shape mismatch {feat.shape}")
            self.combined_static_features = np.concatenate(self.static_features_list, axis=0).astype(np.float32)
            logging.info(f"Combined NON-CLAY static features shape: {self.combined_static_features.shape}")
            logging.info(f"Non-Clay static channels: {self.feature_names}")

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

        # --- Precompute Grids/Masks --- #
        self.grid_coords = compute_grid_cell_coordinates(self.bounds, self.sat_H, self.sat_W)
        self.target_grids, self.valid_masks = precompute_uhi_grids(
            uhi_data=self.uhi_data,
            bounds=self.bounds,
            sat_H=self.sat_H,
            sat_W=self.sat_W,
            resolution_m=self.resolution_m
        )
        self.weather_grids = {} # --- NEW: Dictionary for precomputed weather grids
        self._precompute_weather_grids() # --- NEW: Precompute weather grids
        # --- END Precomputation --- #

        # --- Final Log --- #
        logging.info(f"Dataset initialized for {self.city_name} with {len(self)} unique timestamps.")
        logging.info(f"Target grid size (H, W): ({self.sat_H}, {self.sat_W}), CRS: {self.target_crs_str}")
        logging.info(f"Weather sequence length T = {self.weather_seq_length}")
        logging.info(f"Enabled features (flags): {json.dumps(self.feature_flags)}")
        logging.info(f"Total NON-CLAY static feature channels: {self.combined_static_features.shape[0]}")
        logging.info(f" High-res DEM loaded: {self.high_res_dem_full is not None} (shape: {self.high_res_dem_full.shape if self.high_res_dem_full is not None else 'N/A'})")
        logging.info(f" High-res DSM loaded: {self.high_res_dsm_full is not None} (shape: {self.high_res_dsm_full.shape if self.high_res_dsm_full is not None else 'N/A'})")

    def _precompute_weather_grids(self):
        """Precomputes the weather grid for every unique timestamp using util function."""
        logging.info("Precomputing weather grids for all unique timestamps...")
        for timestamp in tqdm(self.unique_timestamps, desc="Precomputing weather grids"):
            self.weather_grids[timestamp] = build_weather_grid(
                timestamp=timestamp,
                bronx_weather=self.bronx_weather,
                manhattan_weather=self.manhattan_weather,
                bronx_coords=self.bronx_coords,
                manhattan_coords=self.manhattan_coords,
                grid_coords=self.grid_coords,
                sat_H=self.sat_H,
                sat_W=self.sat_W
            )
        logging.info("Finished precomputing weather grids.")

    def __len__(self):
        return len(self.unique_timestamps)

    def __getitem__(self, idx) -> Dict[str, Any]:
        target_timestamp = self.unique_timestamps[idx]

        # --- Retrieve Precomputed UHI Data --- #
        target_grid = self.target_grids[target_timestamp]
        valid_mask = self.valid_masks[target_timestamp]

        # --- Prepare Weather Sequence --- #
        weather_sequence_list = []
        current_ts_index = self.unique_timestamps.index(target_timestamp)
        start_index = max(0, current_ts_index - self.weather_seq_length + 1)

        # Retrieve weather grids for the sequence
        for i in range(start_index, current_ts_index + 1):
            ts = self.unique_timestamps[i]
            weather_grid = self.weather_grids.get(ts)
            if weather_grid is None:
                # Should not happen if precomputation is done, but as fallback:
                logging.warning(f"Precomputed weather grid missing for {ts}. Building on the fly.")
                weather_grid = build_weather_grid(ts, self.bronx_weather, self.manhattan_weather, self.bronx_coords, self.manhattan_coords, self.grid_coords, self.sat_H, self.sat_W)
            weather_sequence_list.append(weather_grid)

        # Pad if sequence is shorter than required (start of dataset)
        num_missing = self.weather_seq_length - len(weather_sequence_list)
        if num_missing > 0:
            padding_grid = weather_sequence_list[0] # Use the earliest available grid for padding
            weather_sequence_list = [padding_grid] * num_missing + weather_sequence_list

        # Stack into sequence: (T, C_weather, H, W)
        weather_seq = np.stack(weather_sequence_list, axis=0)

        # --- Prepare Static Features (Non-Clay) --- #
        static_features = self.combined_static_features # Already combined (C_static_non_clay, H, W)

        # --- Prepare Clay Inputs (if needed) --- #
        cloudless_mosaic_for_clay = None
        norm_latlon_tensor = None
        norm_time_tensor = None
        if self.feature_flags["use_clay"]:
            if self.cloudless_mosaic_full is None:
                # This case should be prevented by checks in __init__
                raise RuntimeError("Clay features enabled, but cloudless mosaic was not loaded.")
            # Clay takes specific bands (e.g., RGB+NIR)
            clay_input_band_names = ["blue", "green", "red", "nir"] # Assuming these are needed
            clay_input_indices = []
            try:
                for band_name in clay_input_band_names:
                    clay_input_indices.append(DEFAULT_MOSAIC_BANDS_ORDER.index(band_name))
                cloudless_mosaic_for_clay = self.cloudless_mosaic_full[clay_input_indices, :, :]
            except ValueError as e:
                raise ValueError(f"Cannot extract required bands for Clay ('{e}') from loaded mosaic bands.")

            # Get static lat/lon embedding (cached)
            norm_latlon_tensor = normalize_clay_latlon(self.bounds)
            # Get dynamic time embedding for the *target* timestamp
            norm_time_tensor = normalize_clay_timestamp(target_timestamp)

        # --- Assemble Sample Dictionary --- #
        sample = {
            'weather_seq': torch.from_numpy(weather_seq).float(), # (T, C_weather, H, W)
            'target': torch.from_numpy(target_grid).float().unsqueeze(0), # (1, H, W)
            'mask': torch.from_numpy(valid_mask).bool().unsqueeze(0),     # (1, H, W) - Use bool for masks
        }

        # Add optional non-clay static features
        if static_features.shape[0] > 0:
            sample['static_features'] = torch.from_numpy(static_features).float()

        # Add optional Clay inputs
        if self.feature_flags["use_clay"]:
            if cloudless_mosaic_for_clay is None or norm_latlon_tensor is None or norm_time_tensor is None:
                 raise RuntimeError("Clay inputs requested but could not be prepared.") # Should not happen
            sample['cloudless_mosaic'] = torch.from_numpy(cloudless_mosaic_for_clay).float()
            sample['norm_latlon'] = torch.from_numpy(norm_latlon_tensor).float()
            sample['norm_timestamp'] = torch.from_numpy(norm_time_tensor).float()

        # --- ADD High-Res DEM/DSM if loaded --- #
        if self.feature_flags.get("use_dem_high_res", False) and self.high_res_dem_full is not None:
            sample['high_res_dem'] = torch.from_numpy(self.high_res_dem_full).float()
        if self.feature_flags.get("use_dsm_high_res", False) and self.high_res_dsm_full is not None:
            sample['high_res_dsm'] = torch.from_numpy(self.high_res_dsm_full).float()
        # --- END ADD --- #

        return sample 