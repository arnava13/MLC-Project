import math
from typing import List, Tuple
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

# --- Import centralized utils --- #
from .data_utils import (
    determine_target_grid_size,
    compute_grid_cell_coordinates,
    precompute_uhi_grids,
    load_process_elevation, # Using the flexible version
    normalize_lst,
    get_closest_weather_data,
    build_weather_grid,
    get_time_embedding,
    normalize_clay_timestamp,
    normalize_clay_latlon,
    WEATHER_NORM_PARAMS # Import constant if needed directly, though build_weather_grid uses it
)
# ------------------------------ #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CityDataSet(Dataset):
    """
    Dataset for UHI modeling using locally stored data (CNN version).
    Uses centralized data utility functions for processing.
    """

    def __init__(self, bounds: List[float], averaging_window, # averaging_window seems unused now?
                 resolution_m: int,
                 uhi_csv: str,
                 bronx_weather_csv: str, manhattan_weather_csv: str,
                 cloudless_mosaic_path: str,
                 data_dir: str, city_name: str,
                 include_lst: bool = True,
                 single_lst_median_path: str = None,
                 dem_tile_dir: str = None, # Keep for backward compatibility if needed, but use load_process_elevation
                 dsm_tile_dir: str = None, # Keep for backward compatibility if needed, but use load_process_elevation
                 elevation_nodata: float = -3.4028235e+38,
                 target_crs: str = "EPSG:4326"
                 ):
        """
        Initialize the dataset.

        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]. REQUIRED.
            averaging_window: Unused parameter? (originally for dynamic LST loading).
            resolution_m: Target spatial resolution (meters).
            uhi_csv: Path to UHI data CSV file.
            bronx_weather_csv: Path to Bronx weather station CSV file.
            manhattan_weather_csv: Path to Manhattan weather station CSV file.
            cloudless_mosaic_path: Path to the pre-generated cloudless Sentinel-2 mosaic (.npy).
            data_dir: Base directory for stored data.
            city_name: Name of the city.
            include_lst: Whether to include Land Surface Temperature data.
            single_lst_median_path (str, optional): Path to a pre-generated single LST median .npy file.
            dem_tile_dir (str, optional): Path to the directory containing DEM GeoTIFF tiles (passed to load_process_elevation).
            dsm_tile_dir (str, optional): Path to the directory containing DSM GeoTIFF tiles (passed to load_process_elevation).
            elevation_nodata (float, optional): Nodata value used in DEM/DSM tiles.
            target_crs (str): The target Coordinate Reference System (e.g., 'EPSG:4326').
        """
        assert bounds and len(bounds) == 4, "Bounds [min_lon, min_lat, max_lon, max_lat] must be provided."
        self.bounds = bounds
        self.include_lst = include_lst
        self.resolution_m = resolution_m
        self.data_dir = Path(data_dir)
        self.city_name = city_name
        self.target_crs_str = target_crs # Store string
        self.target_crs = rasterio.crs.CRS.from_string(target_crs)
        self.elevation_nodata = elevation_nodata

        # --- Load UHI Data --- #
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
        # --- END UHI Load --- #

        # --- Determine Target Grid Size and Transform using Util --- #
        self.sat_H, self.sat_W = determine_target_grid_size(self.bounds, self.resolution_m)
        self.target_transform = rasterio.transform.from_bounds(*self.bounds, self.sat_W, self.sat_H)
        logging.info(f"Target grid size (H, W): ({self.sat_H}, {self.sat_W}), CRS: {self.target_crs_str}")

        # --- Load Static Cloudless Mosaic --- #
        self.cloudless_mosaic_path = Path(cloudless_mosaic_path)
        if not self.cloudless_mosaic_path.exists():
             raise FileNotFoundError(f"Cloudless mosaic file not found: {self.cloudless_mosaic_path}")
        logging.info(f"Loading cloudless mosaic from {self.cloudless_mosaic_path}")
        self.cloudless_mosaic = np.load(self.cloudless_mosaic_path)
        # Resize if necessary
        if self.cloudless_mosaic.shape[1:] != (self.sat_H, self.sat_W):
            zoom_factors = (1, self.sat_H / self.cloudless_mosaic.shape[1], self.sat_W / self.cloudless_mosaic.shape[2])
            logging.info(f"Resizing mosaic from {self.cloudless_mosaic.shape[1:]} to {(self.sat_H, self.sat_W)}")
            self.cloudless_mosaic = zoom(self.cloudless_mosaic, zoom=zoom_factors, order=1)
        self.num_static_bands = self.cloudless_mosaic.shape[0]
        logging.info(f"Loaded mosaic with {self.num_static_bands} bands and shape {self.cloudless_mosaic.shape[1:]}")

        # --- Load Single Static LST Median (if specified and included) --- #
        self.single_lst_median = None
        if self.include_lst:
            if single_lst_median_path:
                lst_path = Path(single_lst_median_path)
                if lst_path.exists():
                    logging.info(f"Loading single LST median from: {lst_path}")
                    try:
                        lst_tensor = np.load(lst_path)
                        if lst_tensor.ndim == 2: lst_tensor = lst_tensor[np.newaxis, :, :]
                        # Resize LST to match target grid shape
                        if lst_tensor.shape[1:] != (self.sat_H, self.sat_W):
                            zoom_factors = (1, self.sat_H / lst_tensor.shape[1], self.sat_W / lst_tensor.shape[2])
                            logging.info(f"Resizing LST from {lst_tensor.shape[1:]} to {(self.sat_H, self.sat_W)}")
                            lst_tensor = zoom(lst_tensor, zoom=zoom_factors, order=1)
                        # Normalize using util function
                        self.single_lst_median = normalize_lst(lst_tensor, self.sat_H, self.sat_W)
                        logging.info(f"Loaded and normalized single LST median with shape {self.single_lst_median.shape}")
                    except Exception as e:
                        logging.error(f"Failed to load or process single LST median from {lst_path}: {e}")
                        self.include_lst = False
                else:
                    logging.error(f"Provided single LST median path not found: {lst_path}")
                    self.include_lst = False
            else:
                logging.warning("include_lst is True, but no single_lst_median_path provided. LST will not be used.")
                self.include_lst = False

        if not self.include_lst:
             # Ensure placeholder exists if LST is disabled
             self.single_lst_median = np.zeros((1, self.sat_H, self.sat_W), dtype=np.float32)

        # --- Load, Reproject, Normalize DEM/DSM using Util --- #
        self.dem_grid = None
        if dem_tile_dir:
            self.dem_grid = load_process_elevation(
                source_path=dem_tile_dir,
            grid_type="DEM",
                bounds=self.bounds,
                resolution_m=self.resolution_m,
                sat_H=self.sat_H,
                sat_W=self.sat_W,
                target_crs_str=self.target_crs_str,
                target_transform=self.target_transform,
                elevation_nodata=self.elevation_nodata
        )
        self.dsm_grid = None
        if dsm_tile_dir:
            self.dsm_grid = load_process_elevation(
                source_path=dsm_tile_dir,
            grid_type="DSM",
                bounds=self.bounds,
                resolution_m=self.resolution_m,
                sat_H=self.sat_H,
                sat_W=self.sat_W,
                target_crs_str=self.target_crs_str,
                target_transform=self.target_transform,
                elevation_nodata=self.elevation_nodata
        )
        self.include_dem = self.dem_grid is not None
        self.include_dsm = self.dsm_grid is not None
        logging.info(f"DEM included: {self.include_dem}")
        logging.info(f"DSM included: {self.include_dsm}")

        # --- Load Weather Station Data --- #
        self.bronx_weather = pd.read_csv(bronx_weather_csv)
        self.manhattan_weather = pd.read_csv(manhattan_weather_csv)
        # Convert datetimes, localize, handle errors
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
        
        # --- Compute Grid Cell Coordinates using Util --- #
        self.grid_coords = compute_grid_cell_coordinates(self.bounds, self.sat_H, self.sat_W)

        # --- Precompute UHI Grids/Masks using Util --- #
        self.target_grids, self.valid_masks = precompute_uhi_grids(
            uhi_data=self.uhi_data,
            bounds=self.bounds,
            sat_H=self.sat_H,
            sat_W=self.sat_W,
            resolution_m=self.resolution_m
        )

        # --- Precompute Static Clay Lat/Lon Embedding (cached) --- #
        self._cached_norm_latlon = normalize_clay_latlon(self.bounds)

        logging.info(f"Dataset initialized for {self.city_name} with {len(self)} unique timestamps.")

    def __len__(self):
        return len(self.unique_timestamps)

    def __getitem__(self, idx):
        timestamp = self.unique_timestamps[idx]

        # --- Retrieve Precomputed UHI Data --- #
        target = self.target_grids[timestamp]
        mask = self.valid_masks[timestamp]

        # --- Prepare Weather Grid using Util --- #
        weather_grid = build_weather_grid(
            timestamp=timestamp,
            bronx_weather=self.bronx_weather,
            manhattan_weather=self.manhattan_weather,
            bronx_coords=self.bronx_coords,
            manhattan_coords=self.manhattan_coords,
            grid_coords=self.grid_coords,
            sat_H=self.sat_H,
            sat_W=self.sat_W
        )

        # --- Prepare Time Embedding using Util --- #
        time_embedding = get_time_embedding(timestamp, self.sat_H, self.sat_W)

        # --- Prepare Clay Metadata using Utils --- #
        norm_latlon_tensor = self._cached_norm_latlon # Fetch precalculated
        norm_time_tensor = normalize_clay_timestamp(timestamp)

        # --- Assemble Sample Dictionary --- #
        sample = {
            'cloudless_mosaic': torch.from_numpy(self.cloudless_mosaic).float(),
            'weather_grid': torch.from_numpy(weather_grid).float(),
            'target': torch.from_numpy(target).float().unsqueeze(0), # Add channel dim
            'mask': torch.from_numpy(mask).bool().unsqueeze(0),      # Use bool for masks
            'time_embedding': torch.from_numpy(time_embedding).float(),
            'norm_latlon': torch.from_numpy(norm_latlon_tensor).float(),
            'norm_timestamp': torch.from_numpy(norm_time_tensor).float(),
        }

        # Add LST if included
        if self.include_lst:
            sample['lst_median'] = torch.from_numpy(self.single_lst_median).float()

        # Add DEM/DSM if loaded
        if self.include_dem:
            sample['dem'] = torch.from_numpy(self.dem_grid).float()
        if self.include_dsm:
            sample['dsm'] = torch.from_numpy(self.dsm_grid).float()

        return sample