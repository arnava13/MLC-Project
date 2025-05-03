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
# ---------------------------------------- #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CityDataSet(Dataset):
    """
    Dataset for UHI modeling using locally stored data.
    Loads a cloudless mosaic & optionally a single LST median for static features,
    plus dynamic weather/time data from specific weather stations. 
    Returns a dictionary for the model.
    """

    def __init__(self, bounds: List[float], averaging_window,
                 resolution_m: int,
                 uhi_csv: str,
                 bronx_weather_csv: str, manhattan_weather_csv: str,  # New parameters for station data
                 cloudless_mosaic_path: str,
                 data_dir: str, city_name: str,
                 include_lst: bool = True,
                 single_lst_median_path: str = None,
                 # --- UPDATED: DEM/DSM Tile Directories ---
                 dem_tile_dir: str = None, # Directory for DEM tiles
                 dsm_tile_dir: str = None, # Directory for DSM tiles
                 elevation_nodata: float = -3.4028235e+38, # Default nodata for NYC LiDAR
                 target_crs: str = "EPSG:4326" # Default target CRS
                 # ---------------------------------------
                 ):
        """
        Initialize the dataset.

        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]. MUST be provided.
            averaging_window: Days lookback for LST median (used only if single_lst_median_path not provided)
            resolution_m: Target spatial resolution (meters) for UHI/Weather/LST grids.
            uhi_csv: Path to UHI data CSV file.
            bronx_weather_csv: Path to Bronx weather station CSV file.
            manhattan_weather_csv: Path to Manhattan weather station CSV file.
            cloudless_mosaic_path: Path to the pre-generated cloudless Sentinel-2 mosaic (.npy).
            data_dir: Base directory for stored data.
            city_name: Name of the city.
            include_lst: Whether to include Land Surface Temperature data.
            single_lst_median_path (str, optional): Path to a pre-generated single LST median .npy file.
                                                    If provided, ignores averaging_window and dynamic loading.
            dem_tile_dir (str, optional): Path to the directory containing DEM GeoTIFF tiles.
            dsm_tile_dir (str, optional): Path to the directory containing DSM GeoTIFF tiles.
            elevation_nodata (float, optional): Nodata value used in DEM/DSM tiles.
            target_crs (str): The target Coordinate Reference System (e.g., 'EPSG:4326') for resampling.
        """
        # --- Basic Parameters ---
        assert bounds and len(bounds) == 4, "Bounds [min_lon, min_lat, max_lon, max_lat] must be provided."
        self.bounds = bounds
        self.include_lst = include_lst
        self.resolution_m = resolution_m
        self.data_dir = Path(data_dir)
        self.city_name = city_name
        self.sat_files_dir = self.data_dir / self.city_name / "sat_files"
        self.cloudless_mosaic_path = Path(cloudless_mosaic_path)

        if not self.sat_files_dir.exists():
            raise FileNotFoundError(f"Base satellite data directory not found: {self.sat_files_dir}")
        if not self.cloudless_mosaic_path.exists():
             raise FileNotFoundError(f"Cloudless mosaic file not found: {self.cloudless_mosaic_path}")

        # --- Load Static Cloudless Mosaic ---
        logging.info(f"Loading cloudless mosaic from {self.cloudless_mosaic_path}")
        self.cloudless_mosaic = np.load(self.cloudless_mosaic_path)
        self.num_static_bands = self.cloudless_mosaic.shape[0]
        logging.info(f"Loaded mosaic with {self.num_static_bands} bands and shape {self.cloudless_mosaic.shape[1:]}")

        # --- Load UHI Data & Determine Target Grid Size ---
        self.uhi_data = pd.read_csv(uhi_csv)
        # Check for 'datetime' column (case-sensitive)
        timestamp_col_name = 'datetime' # Actual name from uhi.csv
        if timestamp_col_name not in self.uhi_data.columns:
            raise ValueError(f"Timestamp column ('{timestamp_col_name}') not found in {uhi_csv}. Found columns: {self.uhi_data.columns.tolist()}")
        # Convert to datetime, localize to US/Eastern, and store unique sorted timestamps
        uhi_dt_format = '%d-%m-%Y %H:%M' # Format for uhi.csv
        target_timezone = 'US/Eastern'
        try:
            all_timestamps_naive = pd.to_datetime(self.uhi_data[timestamp_col_name], format=uhi_dt_format)
            all_timestamps = all_timestamps_naive.dt.tz_localize(target_timezone, ambiguous='infer')
        except ValueError as e:
             logging.error(f"Error parsing or localizing UHI timestamps with format {uhi_dt_format}: {e}. Trying default parsing.")
             all_timestamps_naive = pd.to_datetime(self.uhi_data[timestamp_col_name], errors='coerce')
             if all_timestamps_naive.isnull().any():
                 raise ValueError(f"Failed to parse some UHI timestamps in {uhi_csv}")
             try:
                # Try localizing even if format parsing failed
                all_timestamps = all_timestamps_naive.dt.tz_localize(target_timezone, ambiguous='infer')
             except Exception as loc_e:
                 raise ValueError(f"Failed to localize UHI timestamps: {loc_e}")

        self.unique_timestamps = sorted(all_timestamps.unique())
        self.uhi_data[timestamp_col_name] = all_timestamps # Update column in dataframe to be tz-aware

        self.sat_H, self.sat_W = self._determine_target_grid_size()
        # Store target transform and CRS based on grid size and bounds
        self.target_transform = rasterio.transform.from_bounds(*self.bounds, self.sat_W, self.sat_H)
        self.target_crs = rasterio.crs.CRS.from_string(target_crs)

        # --- Load Single Static LST Median (if specified and included) ---
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
                            lst_tensor = zoom(lst_tensor, zoom=zoom_factors, order=1)
                        # Normalize the static LST
                        self.single_lst_median = self._normalize_lst(lst_tensor)
                        logging.info(f"Loaded and normalized single LST median with shape {self.single_lst_median.shape}")
                    except Exception as e:
                        logging.error(f"Failed to load or process single LST median from {lst_path}: {e}")
                        # Decide whether to raise error or continue without LST
                        self.include_lst = False # Disable LST if loading failed
                else:
                    logging.error(f"Provided single LST median path not found: {lst_path}")
                    self.include_lst = False # Disable LST
            else:
                # This case (include_lst=True but no path provided) should ideally be handled
                # by running the create_sat_tensor_files script first. For robustness,
                # we could try to find a default named file or disable LST here.
                logging.warning("include_lst is True, but no single_lst_median_path provided. LST will not be used.")
                self.include_lst = False

        if not self.include_lst:
             # Ensure placeholder logic works if LST is disabled
             self.single_lst_median = np.zeros((1, self.sat_H, self.sat_W), dtype=np.float32)

        # --- NEW: Load, Reproject, Normalize DEM/DSM from Tiles --- #
        self.dem_grid = self._load_process_elevation_tiles(
            tile_dir=dem_tile_dir,
            tile_prefix="dem", # Assuming files are named dem_*.tif
            grid_type="DEM",
            nodata_val=elevation_nodata
        )
        self.dsm_grid = self._load_process_elevation_tiles(
            tile_dir=dsm_tile_dir,
            tile_prefix="dsm", # Assuming files are named dsm_*.tif
            grid_type="DSM",
            nodata_val=elevation_nodata
        )
        self.include_dem = self.dem_grid is not None
        self.include_dsm = self.dsm_grid is not None
        # --- END NEW --- #

        # --- Load Weather Station Data ---
        self.bronx_weather = pd.read_csv(bronx_weather_csv)
        self.manhattan_weather = pd.read_csv(manhattan_weather_csv)

        # Convert datetime strings to pandas datetime objects, ensuring aware and correct timezone
        target_timezone = 'US/Eastern'

        # --- Process Bronx --- 
        try:
            # Try default parsing first, raise error if invalid strings found
            dt_naive_or_aware = pd.to_datetime(self.bronx_weather['datetime'], errors='raise')
            
            # Ensure it's localized to the target timezone
            if dt_naive_or_aware.dt.tz is None:
                # If naive, localize it
                self.bronx_weather['datetime'] = dt_naive_or_aware.dt.tz_localize(target_timezone, ambiguous='infer')
            elif str(dt_naive_or_aware.dt.tz) != target_timezone:
                # If aware but different timezone, convert it
                logging.warning(f"Converting Bronx weather timezone from {dt_naive_or_aware.dt.tz} to {target_timezone}")
                self.bronx_weather['datetime'] = dt_naive_or_aware.dt.tz_convert(target_timezone)
            else:
                # If already aware and correct timezone, just assign (no change needed)
                 self.bronx_weather['datetime'] = dt_naive_or_aware

        except Exception as e:
            logging.error(f"Failed to parse or localize Bronx weather datetime: {e}")
            # Raise a more informative error
            raise ValueError("Error processing datetimes in Bronx weather file. Check format and timezone info.") from e

        # Check for NaNs post-processing
        if self.bronx_weather['datetime'].isnull().any():
             # This case should ideally not happen with errors='raise', but check anyway
            logging.warning("Some Bronx weather datetime values are NaT after processing.")

        # --- Process Manhattan (similar logic) --- 
        try:
            dt_naive_or_aware = pd.to_datetime(self.manhattan_weather['datetime'], errors='raise')
            
            if dt_naive_or_aware.dt.tz is None:
                self.manhattan_weather['datetime'] = dt_naive_or_aware.dt.tz_localize(target_timezone, ambiguous='infer')
            elif str(dt_naive_or_aware.dt.tz) != target_timezone:
                logging.warning(f"Converting Manhattan weather timezone from {dt_naive_or_aware.dt.tz} to {target_timezone}")
                self.manhattan_weather['datetime'] = dt_naive_or_aware.dt.tz_convert(target_timezone)
            else:
                 self.manhattan_weather['datetime'] = dt_naive_or_aware

        except Exception as e:
            logging.error(f"Failed to parse or localize Manhattan weather datetime: {e}")
            raise ValueError("Error processing datetimes in Manhattan weather file. Check format and timezone info.") from e

        if self.manhattan_weather['datetime'].isnull().any():
            logging.warning("Some Manhattan weather datetime values are NaT after processing.")

        # Weather stations' coordinates (provided in the EY xlsx)
        self.bronx_coords = (40.872, -73.893)       # (lat, lon) for Bronx
        self.manhattan_coords = (40.767, -73.964)   # (lat, lon) for Manhattan
        
        logging.info(f"Loaded Bronx weather data: {len(self.bronx_weather)} records")
        logging.info(f"Loaded Manhattan weather data: {len(self.manhattan_weather)} records")
        
        # --- Precompute Grid Cell Coordinates and Closest Station Map ---
        self.grid_coords, self.closest_station_map = self._compute_grid_cell_coordinates()

        # --- Precompute UHI Grids/Masks (Store in dictionaries) ---
        self.target_grids = {}
        self.valid_masks = {}
        self._precompute_uhi_grids()
        
        logging.info(f"Dataset initialized for {self.city_name} with {len(self)} unique timestamps. LST included: {self.include_lst}")
        logging.info(f"Target grid size (H, W): ({self.sat_H}, {self.sat_W}) CRS: {self.target_crs}")
        logging.info(f"DEM included: {self.include_dem}")
        logging.info(f"DSM included: {self.include_dsm}")

    # --- REPLACED Helper for DEM/DSM Processing --- #
    def _load_process_elevation_tiles(self, tile_dir: str, tile_prefix: str, grid_type: str, nodata_val: float) -> np.ndarray | None:
        """Loads DEM/DSM tiles, merges, reprojects, resamples, and normalizes them."""
        if not tile_dir:
            logging.info(f"{grid_type} tile directory not provided, skipping.")
            return None

        tile_dir_path = Path(tile_dir)
        if not tile_dir_path.is_dir():
            logging.warning(f"{grid_type} tile directory not found at {tile_dir_path}, skipping.")
            return None

        # Find tiles matching the prefix
        tile_paths = list(tile_dir_path.glob(f'{tile_prefix}_*.tif'))
        if not tile_paths:
            logging.warning(f"No {grid_type} tiles found matching prefix '{tile_prefix}_' in {tile_dir_path}, skipping.")
            return None

        logging.info(f"Found {len(tile_paths)} {grid_type} tiles in {tile_dir_path}. Merging and processing...")

        try:
            # Open tile datasets
            src_files_to_mosaic = [rasterio.open(fp) for fp in tile_paths]

            # Merge tiles - this handles mosaicking and can reproject/resample
            # Note: Merging large numbers of tiles can be memory intensive.
            # Bounds are specified in the target CRS for the output extent.
            # Target resolution is derived from the target transform.
            mosaic, out_trans = merge(
                src_files_to_mosaic,
                bounds=self.bounds, # Target bounds
                res=(self.resolution_m, self.resolution_m), # Target resolution (approx)
                nodata=nodata_val,
                target_aligned_pixels=True, # Align pixels to target bounds
                dst_crs=self.target_crs, # Reproject to target CRS
                # Optionally add precision=7 if needed
            )
            
            # Close the source files
            for src in src_files_to_mosaic:
                src.close()

            # At this point, 'mosaic' should be a numpy array (C, H, W) reprojected to target CRS
            # but might not exactly match self.sat_H, self.sat_W due to merge heuristics.
            # We need to explicitly resample/warp it to the final target grid.

            # Create a memory file for the merged data to use with reproject
            with rasterio.io.MemoryFile() as memfile:
                profile = { # Profile for the merged data before final warp
                    'driver': 'GTiff',
                    'height': mosaic.shape[1],
                    'width': mosaic.shape[2],
                    'count': mosaic.shape[0],
                    'dtype': mosaic.dtype,
                    'crs': self.target_crs,
                    'transform': out_trans, # Transform from merge output
                    'nodata': nodata_val
                }
                with memfile.open(**profile) as dataset:
                    dataset.write(mosaic)
                
                # Now open the memory file and reproject exactly to the target grid
                with memfile.open() as src_dataset:
                    destination = np.zeros((self.sat_H, self.sat_W), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src_dataset, 1), # Reproject band 1
                        destination=destination,
                        src_transform=src_dataset.transform,
                        src_crs=src_dataset.crs,
                        dst_transform=self.target_transform, # Final target grid transform
                        dst_crs=self.target_crs,             # Final target grid CRS
                        dst_nodata=nodata_val,                # Use same nodata for output
                        resampling=Resampling.bilinear
                    )

            # Fill nodata values with 0 after reprojection (or use another strategy)
            # Check for NaN as well, as bilinear resampling might introduce them
            mask = np.isnan(destination) | (destination == nodata_val)
            destination[mask] = 0.0 # Fill nodata/NaN with zero - adjust if needed

            # Add channel dimension
            destination = destination[np.newaxis, :, :]

            # Normalize the final grid (example: min-max to -1 to 1)
            # Using realistic elevation ranges for NYC (approx -10m to 300m for DSM)
            # Adjust min/max based on actual data range if known
            min_elev = -10 if grid_type == "DEM" else -10
            max_elev = 150 if grid_type == "DEM" else 350 # Higher max for DSM (buildings)
            
            norm_01 = (destination - min_elev) / (max_elev - min_elev)
            norm_neg1_pos1 = (norm_01 * 2.0) - 1.0
            normalized_grid = np.clip(norm_neg1_pos1, -1.0, 1.0)

            logging.info(f"Processed {grid_type} final shape: {normalized_grid.shape}")
            return normalized_grid.astype(np.float32)

        except Exception as e:
            logging.error(f"Error processing {grid_type} tiles from {tile_dir_path}: {e}", exc_info=True)
            # Ensure source files are closed even if error occurs mid-process
            if 'src_files_to_mosaic' in locals():
                for src in src_files_to_mosaic:
                    if not src.closed:
                         src.close()
            return None
    # --- END REPLACED Helper --- #

    # --- ADDED Normalization Helpers for Clay Metadata ---
    def _normalize_clay_timestamp(self, date: pd.Timestamp) -> np.ndarray:
        """Normalizes timestamp for Clay encoder input (week + hour sin/cos)."""
        if not isinstance(date, pd.Timestamp):
             raise TypeError(f"Input must be a pandas Timestamp, got {type(date)}")

        # Use isocalendar().week which is standard ISO week number (1-52/53)
        week_of_year = date.isocalendar().week
        hour_of_day = date.hour

        # Normalize week and hour
        norm_week = (week_of_year - 1) / 52.0 # Normalize week to approx [0, 1]
        norm_hour = hour_of_day / 24.0        # Normalize hour to [0, 1)

        # Sin/Cos encoding
        week_sin = np.sin(2 * np.pi * norm_week)
        week_cos = np.cos(2 * np.pi * norm_week)
        hour_sin = np.sin(2 * np.pi * norm_hour)
        hour_cos = np.cos(2 * np.pi * norm_hour)

        # Return as a 4-element array (this is passed *per sample*)
        return np.array([week_sin, week_cos, hour_sin, hour_cos], dtype=np.float32)

    def _normalize_clay_latlon(self) -> np.ndarray:
        """
        Calculates the normalized sin/cos encoding of the *center* lat/lon 
        of the dataset bounds, returning a (4,) numpy array.
        Expected by ClayFeatureExtractor as the `norm_latlon_tensor` input.
        """
        # Use self.bounds calculated during __init__
        min_lon, min_lat, max_lon, max_lat = self.bounds
        
        # Calculate center coordinates
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        
        # Normalize using sin/cos
        lat_rad = center_lat * np.pi / 180
        lon_rad = center_lon * np.pi / 180
        
        # Return as a 4-element array
        return np.array([math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)], dtype=np.float32)

    # --- END ADDED Normalization Helpers ---

    def _determine_target_grid_size(self):
        """Determine the target grid size (H, W) based on self.bounds and resolution_m."""
        # Use self.bounds directly
        min_lon, min_lat, max_lon, max_lat = self.bounds

        deg_per_meter_lat = 1 / 111000
        deg_per_meter_lon = 1 / (111320 * math.cos(math.radians((min_lat + max_lat) / 2)))
        height_deg = max_lat - min_lat
        width_deg = max_lon - min_lon

        H = math.ceil(height_deg / (self.resolution_m * deg_per_meter_lat))
        W = math.ceil(width_deg / (self.resolution_m * deg_per_meter_lon))
        return max(1, H), max(1, W)

    def _compute_grid_cell_coordinates(self):
        """
        Compute the lat/lon coordinates for each grid cell and determine which
        weather station is closest to each cell. Uses self.bounds.

        Returns:
            grid_coords: Array of shape (H, W, 2) containing (lat, lon) for each cell
            closest_station_map: Array of shape (H, W) with 0 for Bronx, 1 for Manhattan
        """
        # Use self.bounds directly
        min_lon, min_lat, max_lon, max_lat = self.bounds

        # Create arrays for lat/lon values across the grid
        lats = np.linspace(max_lat, min_lat, self.sat_H)  # Top to bottom
        lons = np.linspace(min_lon, max_lon, self.sat_W)  # Left to right
        
        # Create 2D grid of coordinates
        grid_coords = np.zeros((self.sat_H, self.sat_W, 2))
        closest_station_map = np.zeros((self.sat_H, self.sat_W), dtype=np.int8)
        
        # Calculate distance to stations for each grid cell
        for i in range(self.sat_H):
            for j in range(self.sat_W):
                lat, lon = lats[i], lons[j]
                grid_coords[i, j] = [lat, lon]
                
                # Calculate squared Euclidean distance to each station
                # (we don't need the actual distance, just which one is closer)
                d_bronx = (lat - self.bronx_coords[0])**2 + (lon - self.bronx_coords[1])**2
                d_manhattan = (lat - self.manhattan_coords[0])**2 + (lon - self.manhattan_coords[1])**2
                
                # 0 for Bronx, 1 for Manhattan
                closest_station_map[i, j] = 1 if d_manhattan < d_bronx else 0
                
        logging.info(f"Computed grid cell coordinates and closest station map")
        logging.info(f"Grid cells assigned to Bronx: {np.sum(closest_station_map == 0)}")
        logging.info(f"Grid cells assigned to Manhattan: {np.sum(closest_station_map == 1)}")
        
        return grid_coords, closest_station_map

    def _precompute_uhi_grids(self):
        """Create target UHI grids and masks for all unique timestamps.
           Stores results in self.target_grids and self.valid_masks dictionaries.
        """
        # Use self.bounds directly
        min_lon, min_lat, max_lon, max_lat = self.bounds
        topleft_lat = max_lat # Latitude decreases downwards
        topleft_lon = min_lon # Longitude increases rightwards

        deg_per_meter_lat = 1 / 111000
        deg_per_meter_lon = 1 / (111320 * math.cos(math.radians((min_lat + max_lat) / 2)))

        x_res_deg = self.resolution_m * deg_per_meter_lon
        y_res_deg = self.resolution_m * deg_per_meter_lat

        # Use correct column names 'Longitude' and 'Latitude'
        # Calculate grid indices once for the whole dataframe
        self.uhi_data['x_grid'] = np.clip(np.floor((self.uhi_data['Longitude'] - topleft_lon) / x_res_deg), 0, self.sat_W - 1).astype(int)
        self.uhi_data['y_grid'] = np.clip(np.floor((topleft_lat - self.uhi_data['Latitude']) / y_res_deg), 0, self.sat_H - 1).astype(int)

        # Group by the actual datetime column
        timestamp_col_name = 'datetime'
        grouped = self.uhi_data.groupby(timestamp_col_name)

        for timestamp, group in tqdm(grouped, desc="Precomputing UHI grids"):
            grid = np.full((self.sat_H, self.sat_W), np.nan, dtype=np.float32)
            mask = np.zeros((self.sat_H, self.sat_W), dtype=bool)
            y_indices = group['y_grid'].values
            x_indices = group['x_grid'].values
            uhi_values = group['UHI Index'].values # Use correct column name
            grid[y_indices, x_indices] = uhi_values
            mask[y_indices, x_indices] = True
            # Store in dictionary with timestamp as key
            self.target_grids[timestamp] = grid
            self.valid_masks[timestamp] = mask

    def _normalize_lst(self, lst_tensor: np.ndarray) -> np.ndarray:
        """Normalizes LST tensor from Kelvin to [-1, 1] using fixed range."""
        if lst_tensor is None or not np.any(lst_tensor):
             # Return zeros if input is None or all zeros
             return np.zeros((1, self.sat_H, self.sat_W), dtype=np.float32)
        lst_min_k = 250.0
        lst_max_k = 330.0
        lst_norm_01 = (lst_tensor - lst_min_k) / (lst_max_k - lst_min_k)
        lst_norm_neg1_pos1 = (lst_norm_01 * 2.0) - 1.0
        return np.clip(lst_norm_neg1_pos1, -1.0, 1.0)

    # --- Weather Normalization Constants ---
    WEATHER_NORM_PARAMS = {
        'air_temp': {'min': -15.0, 'max': 40.0},     # Celsius
        'rel_humidity': {'min': 0.0, 'max': 100.0},  # Percentage
        'avg_windspeed': {'min': 0.0, 'max': 30.0},  # m/s
        'solar_flux': {'min': 0.0, 'max': 1100.0}, # W/m^2
        'wind_direction': {'min': 0.0, 'max': 360.0} # Degrees (for potential direct interpolation before sin/cos)
    }
    # --- End Weather Normalization Constants ---

    def _normalize_min_max(self, value, var_name):
        params = self.WEATHER_NORM_PARAMS.get(var_name)
        if not params:
            return value # No normalization defined
        min_val, max_val = params['min'], params['max']
        # Normalize to [0, 1]
        norm_01 = (value - min_val) / (max_val - min_val)
        # Clip to ensure values stay within [0, 1] after normalization
        return np.clip(norm_01, 0.0, 1.0)

    def _get_closest_weather_data(self, timestamp):
        """
        Get the weather data from both stations for the timestamp closest to the given one.
        
        Args:
            timestamp: Target timestamp to find weather data for
            
        Returns:
            Dictionary with weather data from both stations
        """
        # Find closest timestamp in each station's data
        bronx_idx = (self.bronx_weather['datetime'] - timestamp).abs().idxmin()
        manhattan_idx = (self.manhattan_weather['datetime'] - timestamp).abs().idxmin()
        
        # Get the weather data for those closest timestamps
        bronx_data = self.bronx_weather.loc[bronx_idx]
        manhattan_data = self.manhattan_weather.loc[manhattan_idx]
        
        # Extract relevant columns (5 weather variables)
        weather_vars = ['air_temp', 'rel_humidity', 'avg_windspeed', 'wind_direction', 'solar_flux']
        
        return {
            'bronx': {var: bronx_data[var] for var in weather_vars},
            'manhattan': {var: manhattan_data[var] for var in weather_vars}
        }

    def _build_weather_grid(self, timestamp):
        """
        Build a normalized weather grid using inverse distance weighted (IDW)
        interpolation between the two stations. Wind direction sin/cos components
        are interpolated directly.

        Args:
            timestamp: Target timestamp to get weather data for

        Returns:
            6-channel tensor of shape (6, H, W) with normalized interpolated weather data:
            [temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos, solar_flux]
        """
        # Get weather data from both stations for this timestamp
        station_data = self._get_closest_weather_data(timestamp)
        bronx_raw = station_data['bronx']
        manhattan_raw = station_data['manhattan']

        # Pre-allocate the weather grid (6 channels × H × W)
        weather_grid = np.zeros((6, self.sat_H, self.sat_W), dtype=np.float32)

        # Calculate distances from each grid cell to stations
        # self.grid_coords shape: (H, W, 2) -> (lat, lon)
        # station coords: (lat, lon)
        # Using squared Euclidean distance for simplicity (monotonic with distance)
        lat_grid = self.grid_coords[:, :, 0]
        lon_grid = self.grid_coords[:, :, 1]

        dist_sq_bronx = (lat_grid - self.bronx_coords[0])**2 + (lon_grid - self.bronx_coords[1])**2
        dist_sq_manhattan = (lat_grid - self.manhattan_coords[0])**2 + (lon_grid - self.manhattan_coords[1])**2

        # Inverse distance weighting (power p=2)
        # Add small epsilon to avoid division by zero if a cell is exactly at a station
        epsilon = 1e-9
        weight_bronx = 1.0 / (dist_sq_bronx + epsilon)
        weight_manhattan = 1.0 / (dist_sq_manhattan + epsilon)

        total_weight = weight_bronx + weight_manhattan

        norm_weight_bronx = weight_bronx / total_weight
        norm_weight_manhattan = weight_manhattan / total_weight

        # Interpolate normalized values (except wind direction initially)
        weather_vars_to_interpolate = ['air_temp', 'rel_humidity', 'avg_windspeed', 'solar_flux']
        channel_map = {'air_temp': 0, 'rel_humidity': 1, 'avg_windspeed': 2, 'solar_flux': 5} # Map var name to grid channel index

        for var_name in weather_vars_to_interpolate:
            # Normalize station values
            val_bronx_norm = self._normalize_min_max(bronx_raw[var_name], var_name)
            val_manhattan_norm = self._normalize_min_max(manhattan_raw[var_name], var_name)

            # Interpolate using normalized weights
            interpolated_norm_val = (val_bronx_norm * norm_weight_bronx +
                                     val_manhattan_norm * norm_weight_manhattan)

            channel_idx = channel_map[var_name]
            weather_grid[channel_idx] = interpolated_norm_val

        # --- MODIFIED: Handle Wind Direction ---
        # Convert station angles to sin/cos first
        wd_bronx_deg = bronx_raw['wind_direction']
        wd_manhattan_deg = manhattan_raw['wind_direction']

        wd_bronx_rad = np.deg2rad(wd_bronx_deg)
        wd_manhattan_rad = np.deg2rad(wd_manhattan_deg)

        sin_bronx = np.sin(wd_bronx_rad)
        cos_bronx = np.cos(wd_bronx_rad)
        sin_manhattan = np.sin(wd_manhattan_rad)
        cos_manhattan = np.cos(wd_manhattan_rad)

        # Interpolate sin and cos components using IDW weights
        interpolated_sin = (sin_bronx * norm_weight_bronx +
                            sin_manhattan * norm_weight_manhattan)
        interpolated_cos = (cos_bronx * norm_weight_bronx +
                            cos_manhattan * norm_weight_manhattan)

        # Normalize the resulting vector [sin, cos] to ensure unit length
        length = np.sqrt(interpolated_sin**2 + interpolated_cos**2 + epsilon) # Add epsilon for stability
        norm_interpolated_sin = interpolated_sin / length
        norm_interpolated_cos = interpolated_cos / length

        # Assign to grid channels
        weather_grid[3] = norm_interpolated_sin # Sin component (channel 3)
        weather_grid[4] = norm_interpolated_cos # Cos component (channel 4)
        # --- END MODIFICATION ---

        return weather_grid
    
    def _get_time_embedding(self, timestamp: pd.Timestamp) -> np.ndarray:
        """Computes sin/cos embeddings for minute of day."""
        total_minutes = timestamp.hour * 60 + timestamp.minute
        # day_of_year = timestamp.dayofyear # Removed day of year calculation
        norm_minute = total_minutes / 1440.0
        # norm_day = (day_of_year - 1) / 365.0 # Removed day of year calculation
        minute_sin = np.sin(2 * np.pi * norm_minute)
        minute_cos = np.cos(2 * np.pi * norm_minute)
        # day_sin = np.sin(2 * np.pi * norm_day) # Removed day of year calculation
        # day_cos = np.cos(2 * np.pi * norm_day) # Removed day of year calculation
        time_features = np.array([minute_sin, minute_cos], dtype=np.float32)
        time_map = np.tile(time_features[:, np.newaxis, np.newaxis], (1, self.sat_H, self.sat_W))
        return time_map

    def __len__(self):
        # Length is the number of unique timestamps
        return len(self.unique_timestamps)

    def __getitem__(self, idx):
        # Get the unique timestamp corresponding to the index
        timestamp = self.unique_timestamps[idx]

        # --- Retrieve Precomputed UHI Data ---
        target = self.target_grids[timestamp]
        mask = self.valid_masks[timestamp]

        # --- Prepare Weather Grid ---
        weather_grid = self._build_weather_grid(timestamp)

        # --- Prepare Time Embedding ---
        time_embedding = self._get_time_embedding(timestamp)

        # --- Prepare Clay Metadata (Static) ---
        # Calculate only once if needed, or fetch precalculated
        norm_latlon_tensor = getattr(self, '_cached_norm_latlon', None)
        if norm_latlon_tensor is None:
            norm_latlon_tensor = self._normalize_clay_latlon()
            self._cached_norm_latlon = norm_latlon_tensor
        # Normalize timestamp per sample
        norm_time_tensor = self._normalize_clay_timestamp(timestamp)

        # --- Assemble Sample Dictionary ---
        sample = {
            'cloudless_mosaic': torch.from_numpy(self.cloudless_mosaic).float(),
            'weather_grid': torch.from_numpy(weather_grid).float(),
            'target': torch.from_numpy(target).float().unsqueeze(0), # Add channel dim
            'mask': torch.from_numpy(mask).float().unsqueeze(0),      # Add channel dim
            'time_embedding': torch.from_numpy(time_embedding).float(),
            # Clay specific metadata (match keys expected by ClayFeatureExtractor)
            'norm_latlon': torch.from_numpy(norm_latlon_tensor).float(),
            'norm_timestamp': torch.from_numpy(norm_time_tensor).float(),
        }

        # Add LST if included
        if self.include_lst:
            # Assume single_lst_median is loaded and normalized in __init__
            sample['lst_median'] = torch.from_numpy(self.single_lst_median).float()

        # --- ADD DEM/DSM if loaded ---
        if self.include_dem:
            sample['dem'] = torch.from_numpy(self.dem_grid).float() # Already has channel dim
        if self.include_dsm:
            sample['dsm'] = torch.from_numpy(self.dsm_grid).float() # Already has channel dim
        # --- END ADD ---

        return sample