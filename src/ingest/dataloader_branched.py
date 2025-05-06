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

class CityDataSetBranched(Dataset):
    """
    Dataset for UHI modeling using locally stored data, adapted for branched model.
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
                 weather_seq_length: int = 60,
                 target_crs_str: str = "EPSG:4326",
                 # --- Other Params --- #
                 ): # Removed low-res elev path/nodata args
        """
        Initialize the branched dataset with common feature resampling.

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
            weather_seq_length (int): Number of weather time steps (Default: 60).
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
        self.weather_seq_length = weather_seq_length

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
            # USE mmap_mode='r'
            self.cloudless_mosaic_full_np = np.load(mosaic_path, mmap_mode='r')
            # Determine mosaic transform (assuming it matches UHI grid originally)
            mosaic_h_orig, mosaic_w_orig = self.cloudless_mosaic_full_np.shape[1:]
            self.mosaic_transform = rasterio.transform.from_bounds(*self.bounds, mosaic_w_orig, mosaic_h_orig)
            logging.info(f"Loaded mosaic shape (native res): {self.cloudless_mosaic_full_np.shape}")

        # 4. LST Median (Load with rioxarray, keep as object)
        self.lst_xr = None
        if self.feature_flags["use_lst"]:
            if not self._single_lst_median_path: raise ValueError("single_lst_median_path required if use_lst is True.")
            lst_path = Path(self._single_lst_median_path)
            if lst_path.exists():
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
                else: # Path doesn't exist
                    logging.warning(f"LST path specified but not found: {self._single_lst_median_path}")

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

        # --- Precompute Weather Grid Coordinates (at FEATURE resolution) --- #
        self.weather_grid_coords = compute_grid_cell_coordinates(self.bounds, self.feat_H, self.feat_W)
        # --- Precompute Static Clay Lat/Lon Embedding --- #
        self._cached_norm_latlon = normalize_clay_latlon(self.bounds)

        # --- Final Log --- #
        logging.info(f"Dataset initialized for {self.city_name} with {len(self)} unique timestamps.")
        logging.info(f"Weather sequence length T = {self.weather_seq_length}")
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

        # --- Prepare Weather Sequence (at FEATURE resolution) --- #
        weather_sequence_list = []
        current_ts_index = self.unique_timestamps.index(target_timestamp)
        start_index = max(0, current_ts_index - self.weather_seq_length + 1)

        for i in range(start_index, current_ts_index + 1):
            ts = self.unique_timestamps[i]
            # Build weather grid AT FEATURE RESOLUTION
            weather_grid_ts = build_weather_grid(
                timestamp=ts,
                bronx_weather=self.bronx_weather,
                manhattan_weather=self.manhattan_weather,
                bronx_coords=self.bronx_coords,
                manhattan_coords=self.manhattan_coords,
                grid_coords=self.weather_grid_coords, # Use coords for feature grid
                sat_H=self.feat_H, # Use feature grid dimensions
                sat_W=self.feat_W
            )
            weather_sequence_list.append(weather_grid_ts)

        # Pad if sequence is shorter
        num_missing = self.weather_seq_length - len(weather_sequence_list)
        if num_missing > 0:
            padding_grid = weather_sequence_list[0]
            weather_sequence_list = [padding_grid] * num_missing + weather_sequence_list

        weather_seq_feat_res = np.stack(weather_sequence_list, axis=0)

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
            # Wrap numpy mosaic in xarray for resampling
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
                needs_mosaic = False # Disable downstream use if resampling failed
            else:
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
                            logging.warning(f"Requested composite band '{band_name}' not found. Skipping.")
                    if selected_indices:
                        mosaic_subset = mosaic_feat_res[selected_indices, :, :]
                        static_features_list.append(mosaic_subset)
                        feature_names.extend([f"sentinel_{b}" for b in selected_band_names_ordered])

                # Calculate spectral indices if flagged (using RESAMPLED mosaic)
                available_bands_in_resampled = {band: i for i, band in enumerate(DEFAULT_MOSAIC_BANDS_ORDER[:mosaic_feat_res.shape[0]])}
                def _calculate_index(index_name, band_num_name, band_den_name):
                    num_idx = available_bands_in_resampled.get(band_num_name)
                    den_idx = available_bands_in_resampled.get(band_den_name)
                    if num_idx is None or den_idx is None:
                        logging.warning(f"Cannot calculate {index_name.upper()} at feature res: Required bands not available.")
                        return None
                    numerator_band = mosaic_feat_res[num_idx].astype(np.float32)
                    denominator_band = mosaic_feat_res[den_idx].astype(np.float32)
                    denominator_sum = numerator_band + denominator_band
                    index_map = np.full(denominator_sum.shape, 0.0, dtype=np.float32)
                    valid_mask = np.abs(denominator_sum) > 1e-6
                    index_map[valid_mask] = (numerator_band[valid_mask] - denominator_band[valid_mask]) / denominator_sum[valid_mask]
                    index_map = np.clip(index_map, -1.0, 1.0)
                    return index_map[np.newaxis, :, :]

                # ONLY add indices if their corresponding feature flags are TRUE
                if self.feature_flags["use_ndvi"]:
                    ndvi_map = _calculate_index("ndvi", "nir", "red")
                    if ndvi_map is not None: 
                        static_features_list.append(ndvi_map)
                        feature_names.append("ndvi")
                if self.feature_flags["use_ndbi"]:
                    ndbi_map = _calculate_index("ndbi", "swir16", "nir")
                    if ndbi_map is not None: 
                        static_features_list.append(ndbi_map)
                        feature_names.append("ndbi")
                if self.feature_flags["use_ndwi"]:
                    ndwi_map = _calculate_index("ndwi", "green", "nir")
                    if ndwi_map is not None: 
                        static_features_list.append(ndwi_map)
                        feature_names.append("ndwi")

        # 2. LST Median (Resample to FEATURE resolution)
        lst_feat_res = None
        if self.feature_flags["use_lst"] and self.lst_xr is not None:
            lst_feat_res_raw = resample_xarray_to_target(
                self.lst_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs, fill_value=0.0 # Fill with 0
            )
            if lst_feat_res_raw is not None:
                lst_feat_res = normalize_lst(lst_feat_res_raw, self.feat_H, self.feat_W)
                static_features_list.append(lst_feat_res)
                feature_names.append("lst")
            else:
                logging.warning(f"LST resampling failed for timestamp {target_timestamp}.")

        # 3. DEM (Resample to FEATURE resolution)
        dem_feat_res = None
        if self.feature_flags["use_dem"] and self.dem_xr is not None:
            dem_feat_res = resample_xarray_to_target(
                self.dem_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs, fill_value=0.0 # Fill nodata with 0
            )
            if dem_feat_res is not None:
                # Normalize DEM [0, 1] - specific to this feature resolution grid
                min_v, max_v = np.min(dem_feat_res), np.max(dem_feat_res)
                if max_v > min_v:
                    dem_feat_res = (dem_feat_res - min_v) / (max_v - min_v)
                else:
                    dem_feat_res.fill(0.5)
                static_features_list.append(dem_feat_res)
                feature_names.append("dem")
            else:
                logging.warning(f"DEM resampling failed for timestamp {target_timestamp}.")

        # 4. DSM (Resample to FEATURE resolution)
        dsm_feat_res = None
        if self.feature_flags["use_dsm"] and self.dsm_xr is not None:
            dsm_feat_res = resample_xarray_to_target(
                self.dsm_xr, self.feat_H, self.feat_W, self.feat_transform, self.target_crs, fill_value=0.0
            )
            if dsm_feat_res is not None:
                # Normalize DSM [0, 1]
                min_v, max_v = np.min(dsm_feat_res), np.max(dsm_feat_res)
                if max_v > min_v:
                    dsm_feat_res = (dsm_feat_res - min_v) / (max_v - min_v)
                else:
                    dsm_feat_res.fill(0.5)
                static_features_list.append(dsm_feat_res)
                feature_names.append("dsm")
            else:
                logging.warning(f"DSM resampling failed for timestamp {target_timestamp}.")

        # Combine ALL static features (excluding Clay mosaic input)
        if not static_features_list:
            # Need to handle case where no static features are available/enabled
            combined_static_features = np.zeros((0, self.feat_H, self.feat_W), dtype=np.float32)
        else:
            # Ensure all have the correct shape before concatenating
            valid_static_features = []
            valid_feature_names = []
            for i, feat in enumerate(static_features_list):
                if feat is not None and feat.shape[1:] == (self.feat_H, self.feat_W):
                    valid_static_features.append(feat)
                    valid_feature_names.append(feature_names[i])
                else:
                    logging.warning(f"Static feature '{feature_names[i]}' has incorrect shape or is None. Skipping.")
            if valid_static_features:
                combined_static_features = np.concatenate(valid_static_features, axis=0).astype(np.float32)
                logging.debug(f"Included static features: {valid_feature_names}")
            else:
                combined_static_features = np.zeros((0, self.feat_H, self.feat_W), dtype=np.float32)

        # <<< DEBUG PRINT >>>
        print(f"[DEBUG DATALOADER] combined_static_features shape: {combined_static_features.shape}")
        # <<< END DEBUG >>>

        # --- Prepare Clay Inputs (if needed) --- #
        clay_mosaic_input = None # The mosaic resampled to feature res
        norm_latlon_tensor = None
        norm_time_tensor = None
        if self.feature_flags["use_clay"]:
            if mosaic_feat_res is None:
                logging.warning("Clay features enabled, but mosaic could not be loaded/resampled. Skipping Clay.")
            else:
            # Clay takes specific bands (e.g., RGB+NIR)
                clay_input_band_names = ["blue", "green", "red", "nir"]
                clay_input_indices = []
                available_bands_in_resampled = {band: i for i, band in enumerate(DEFAULT_MOSAIC_BANDS_ORDER[:mosaic_feat_res.shape[0]])}
                try:
                    for band_name in clay_input_band_names:
                        clay_input_indices.append(available_bands_in_resampled[band_name])
                    clay_mosaic_input = mosaic_feat_res[clay_input_indices, :, :]
                    norm_latlon_tensor = self._cached_norm_latlon
                    norm_time_tensor = normalize_clay_timestamp(target_timestamp)
                except KeyError as e:
                    logging.warning(f"Cannot extract required bands for Clay ('{e}') from resampled mosaic bands. Skipping Clay.")
                    clay_mosaic_input = None

        # --- Assemble Sample Dictionary --- #
        sample = {
            'weather_seq': torch.from_numpy(weather_seq_feat_res).float(), # (T, C_weather, H_feat, W_feat)
            'target': torch.from_numpy(target_grid).float().unsqueeze(0),   # (1, H_uhi, W_uhi)
            'mask': torch.from_numpy(valid_mask).bool().unsqueeze(0),       # (1, H_uhi, W_uhi)
        }

        # Add combined static features (LST, DEM, DSM, indices, composite)
        if combined_static_features.shape[0] > 0:
            sample['static_features'] = torch.from_numpy(combined_static_features).float()

        # Add optional Clay inputs
        if self.feature_flags["use_clay"] and clay_mosaic_input is not None:
            sample['cloudless_mosaic'] = torch.from_numpy(clay_mosaic_input).float()
            sample['norm_latlon'] = torch.from_numpy(norm_latlon_tensor).float()
            sample['norm_timestamp'] = torch.from_numpy(norm_time_tensor).float()

        return sample 

    def __del__(self):
        if hasattr(self, 'dem_xr') and self.dem_xr is not None:
            try:
                self.dem_xr.close()
                logging.debug("Closed DEM file handle.")
            except Exception as e:
                logging.warning(f"Exception closing DEM file handle: {e}")
        if hasattr(self, 'dsm_xr') and self.dsm_xr is not None:
            try:
                self.dsm_xr.close()
                logging.debug("Closed DSM file handle.")
            except Exception as e:
                logging.warning(f"Exception closing DSM file handle: {e}")
        if hasattr(self, 'lst_xr') and self.lst_xr is not None:
            try:
                self.lst_xr.close()
                logging.debug("Closed LST file handle.")
            except Exception as e:
                logging.warning(f"Exception closing LST file handle: {e}") 