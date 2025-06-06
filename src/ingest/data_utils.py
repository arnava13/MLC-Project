import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.ndimage import zoom
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import rasterio.io
import requests
import os
import subprocess
import time
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Weather Normalization Constants & Feature Definitions --- #
WEATHER_VARIABLES_INFO = {
    'air_temp': {'channels': 1, 'norm_params': {'min': -15.0, 'max': 40.0}}, # Celsius
    'rel_humidity': {'channels': 1, 'norm_params': {'min': 0.0, 'max': 100.0}}, # Percentage
    'avg_windspeed': {'channels': 1, 'norm_params': {'min': 0.0, 'max': 30.0}}, # m/s
    'solar_flux': {'channels': 1, 'norm_params': {'min': 0.0, 'max': 1100.0}}, # W/m^2
    'wind_direction': {'channels': 2, 'norm_params': {'min': 0.0, 'max': 360.0}} # Degrees, becomes sin/cos
}

# Order in which channels will be stacked if all are enabled.
# This defines the canonical order for indexing in build_weather_grid.
CANONICAL_WEATHER_FEATURE_ORDER = [
    'air_temp',       # Channel 0
    'rel_humidity',   # Channel 1
    'avg_windspeed',  # Channel 2
    'wind_direction_sin',   # Channel 3 (derived from wind_direction)
    'wind_direction_cos',   # Channel 4 (derived from wind_direction)
    'solar_flux'      # Channel 5
]

def calculate_actual_weather_channels(enabled_features: List[str]) -> int:
    """Calculates the number of grid channels based on enabled weather features."""
    count = 0
    for feature_name in enabled_features:
        if feature_name not in WEATHER_VARIABLES_INFO:
            # This case should ideally be caught by config validation earlier
            logging.warning(f"Weather feature '{feature_name}' not recognized in WEATHER_VARIABLES_INFO. Skipping.")
            continue
        count += WEATHER_VARIABLES_INFO[feature_name]['channels']
    return count

# --- Grid/Coordinate Utilities --- #

def determine_target_grid_size(bounds: List[float], resolution_m: int) -> Tuple[int, int]:
    """Determine target grid H, W based on bounds and resolution."""
    min_lon, min_lat, max_lon, max_lat = bounds
    # Approx degrees per meter
    deg_per_meter_lat = 1 / 111000
    deg_per_meter_lon = 1 / (111320 * math.cos(math.radians((min_lat + max_lat) / 2)))
    height_deg = max_lat - min_lat
    width_deg = max_lon - min_lon
    H = math.ceil(height_deg / (resolution_m * deg_per_meter_lat))
    W = math.ceil(width_deg / (resolution_m * deg_per_meter_lon))
    return max(1, H), max(1, W)

def compute_grid_cell_coordinates(bounds: List[float], sat_H: int, sat_W: int, target_crs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon coordinates for each target grid cell center."""
    min_lon, min_lat, max_lon, max_lat = bounds
    # Calculate pixel size in degrees
    x_res = (max_lon - min_lon) / sat_W
    y_res = (max_lat - min_lat) / sat_H
    # Calculate cell center coordinates
    lons = np.linspace(min_lon + x_res/2, max_lon - x_res/2, sat_W)
    lats = np.linspace(max_lat - y_res/2, min_lat + y_res/2, sat_H) # Lat decreases downwards
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    logging.info(f"Computed grid cell center coordinates for CRS: {target_crs}.")
    return grid_lon, grid_lat

# --- Elevation Data Processing --- #

def load_process_elevation(
    elevation_path: str,
    data_source_name: str, # e.g., "DEM" or "DSM"
    bounds: List[float],
    resolution_m: int,
    target_H: int,
    target_W: int,
    target_crs_str: str,
    target_transform: Any, # rasterio transform object
    elevation_nodata: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    Loads, merges (if needed), reprojects, fills nodata, normalizes,
    and resizes DEM/DSM data to match the target grid.

    Handles both single TIF file paths and directories containing TIF tiles.

    Args:
        elevation_path: Path to the DEM/DSM data (directory or single .tif file).
        data_source_name: Name for logging ("DEM" or "DSM").
        bounds: Target bounding box [min_lon, min_lat, max_lon, max_lat].
        resolution_m: Target spatial resolution in meters.
        target_H: Target height in pixels.
        target_W: Target width in pixels.
        target_crs_str: Target CRS string (e.g., "EPSG:4326").
        target_transform: Target rasterio Affine transform.
        elevation_nodata: Nodata value in the source file(s). If None, uses defaults for known NYC data.

    Returns:
        A numpy array (1, H, W) with processed elevation data, or None on failure.
    """
    elevation_path_obj = Path(elevation_path)
    target_crs = rasterio.crs.CRS.from_string(target_crs_str)

    # --- Determine Nodata Value --- #
    nodata_value = elevation_nodata
    if nodata_value is None:
        if data_source_name == "DEM":
            nodata_value = -3.4028234663852886e+38 # Default for NYS DEM
            logging.info(f"Using default nodata value {nodata_value} for {data_source_name}")
        elif data_source_name == "DSM":
            nodata_value = -9999.0 # Default for NYC OpenData DSM
            logging.info(f"Using default nodata value {nodata_value} for {data_source_name}")
        else:
            # This case should ideally be prevented by checks in the Dataset __init__ if defaults are not used.
            logging.error(f"elevation_nodata is None and no default known for {data_source_name}. Cannot proceed.")
            return None

    merged_data = None
    source_crs = None
    source_transform = None

    try:
        if elevation_path_obj.is_dir():
            # --- Handle Directory of Tiles --- #
            tile_paths = list(elevation_path_obj.glob('*.tif'))
            if not tile_paths:
                logging.warning(f"No {data_source_name} tiles (*.tif) found in directory {elevation_path_obj}, skipping.")
                return None
            logging.info(f"Found {len(tile_paths)} {data_source_name} tiles in {elevation_path_obj}. Merging..." )

            src_files_to_mosaic = []
            try:
                src_files_to_mosaic = [rasterio.open(fp) for fp in tile_paths]
                first_crs = src_files_to_mosaic[0].crs
                if not all(src.crs == first_crs for src in src_files_to_mosaic):
                    logging.warning(f"Inconsistent CRSs found among {data_source_name} tiles. Using CRS of first tile ({first_crs}).")
                merged_crs = first_crs # Store the CRS of the source tiles

                mosaic, out_trans = merge(
                    src_files_to_mosaic,
                    bounds=bounds,
                    res=(resolution_m, resolution_m),
                    nodata=nodata_value,
                    target_aligned_pixels=True,
                    dst_crs=target_crs, # Reproject during merge
                    resampling=Resampling.bilinear
                )
                merged_data = mosaic
                merged_transform = out_trans
                merged_crs = target_crs # CRS after merge is target CRS
                logging.info(f"Merged {data_source_name} tiles. Shape: {merged_data.shape}")
            finally:
                for src in src_files_to_mosaic:
                     src.close()

        elif elevation_path_obj.is_file() and elevation_path_obj.suffix.lower() == '.tif':
             # --- Handle Single File --- #
             logging.info(f"Loading single {data_source_name} file from {elevation_path_obj}")
             with rasterio.open(elevation_path_obj) as src:
                 merged_crs = src.crs
                 merged_nodata = src.nodata if src.nodata is not None else nodata_value
                 merged_data = src.read()
                 merged_transform = src.transform
                 logging.info(f"Loaded single {data_source_name} file. Shape: {merged_data.shape}, CRS: {merged_crs}")
        else:
             logging.warning(f"Invalid path for {data_source_name}: {elevation_path_obj}. Must be a directory of .tif files or a single .tif file. Skipping.")
             return None

        # --- Reprojection & Resampling to Target Grid --- #
        if merged_data is None:
             logging.warning(f"No {data_source_name} data loaded. Skipping final processing.")
             return None

        logging.info(f"Warping {data_source_name} to target grid ({target_H}x{target_W}) CRS: {target_crs_str}")
        destination = np.zeros((target_H, target_W), dtype=np.float32)

        if merged_data.ndim == 2:
             merged_data = merged_data[np.newaxis, :, :]
        elif merged_data.ndim != 3:
             raise ValueError(f"Unexpected shape for loaded {data_source_name} data: {merged_data.shape}")

        with rasterio.io.MemoryFile() as memfile:
            profile = {
                'driver': 'GTiff', 'height': merged_data.shape[1], 'width': merged_data.shape[2],
                'count': merged_data.shape[0], 'dtype': merged_data.dtype, 'crs': merged_crs,
                'transform': merged_transform, 'nodata': merged_nodata
            }
            with memfile.open(**profile) as dataset:
                dataset.write(merged_data)

            with memfile.open() as source_data:
                reproject(
                    source=source_data,
                    destination=destination,
                    src_transform=source_data.transform,
                    src_crs=source_data.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=nodata_value, # Use determined nodata value
                    dst_nodata=np.nan # Use NaN internally for easier handling
                )
                reprojected_grid = destination[0]

                # --- Fill NaN Nodata values (introduced by reprojection or present in source) --- #
                # Use mean of valid neighbors or global mean if needed
                if np.isnan(reprojected_grid).any():
                    valid_mask = ~np.isnan(reprojected_grid)
                    if valid_mask.sum() > 0:
                         grid_mean = np.mean(reprojected_grid[valid_mask])
                         reprojected_grid[np.isnan(reprojected_grid)] = grid_mean
                         logging.info(f"Filled NaN nodata values in {data_source_name} with mean {grid_mean:.2f}")
                    else:
                         logging.warning(f"No valid pixels found in {data_source_name} after reprojection. Filling with 0.")
                         reprojected_grid.fill(0)

                # --- Normalization (Min-Max to [0, 1]) --- #
                min_val = np.min(reprojected_grid)
                max_val = np.max(reprojected_grid)
                if max_val > min_val:
                    normalized_grid = (reprojected_grid - min_val) / (max_val - min_val)
                    logging.info(f"Normalized {data_source_name} to [0, 1] (Min: {min_val:.2f}, Max: {max_val:.2f})")
                elif max_val == min_val:
                     # Handle case where all values are the same (flat terrain/surface)
                     normalized_grid = np.full_like(reprojected_grid, 0.5) # Assign a mid-value (e.g., 0.5)
                     logging.info(f"Normalized {data_source_name} to 0.5 (all values were {min_val:.2f})")
                else: # Should not happen if nodata filling worked
                    logging.warning(f"Could not normalize {data_source_name}, max <= min after nodata fill. Using raw grid.")
                    normalized_grid = reprojected_grid

                # Ensure final shape is correct (just in case reprojection slightly off)
                if normalized_grid.shape != (target_H, target_W):
                    logging.warning(f"Resizing {data_source_name} from {normalized_grid.shape} to {(target_H, target_W)} post-reprojection.")
                    # Use zoom for resizing
                    zoom_factors = (target_H / normalized_grid.shape[0], target_W / normalized_grid.shape[1])
                    final_grid = zoom(normalized_grid, zoom=zoom_factors, order=1, mode='bicubic')
                else:
                    final_grid = normalized_grid

                # Add channel dimension
                return final_grid[np.newaxis, :, :].astype(np.float32)

    except Exception as e:
        logging.error(f"Error processing {data_source_name} from {elevation_path_obj}: {e}", exc_info=True)
        return None


# --- UHI/LST Utilities --- #
def precompute_uhi_grids(uhi_data: pd.DataFrame,
                         bounds: List[float],
                         sat_H: int, sat_W: int,
                         resolution_m: int) -> Tuple[Dict[pd.Timestamp, np.ndarray], Dict[pd.Timestamp, np.ndarray]]:
    # Create target UHI grids and masks for all unique timestamps in the dataframe
    target_grids = {}
    valid_masks = {}

    min_lon, min_lat, max_lon, max_lat = bounds
    topleft_lat = max_lat
    topleft_lon = min_lon
    x_res_deg = (max_lon - min_lon) / sat_W
    y_res_deg = (max_lat - min_lat) / sat_H # Note: lat decreases, so y increases

    if 'Longitude' not in uhi_data.columns or 'Latitude' not in uhi_data.columns:
        raise ValueError("UHI data must contain 'Longitude' and 'Latitude' columns.")
    if 'UHI Index' not in uhi_data.columns:
         raise ValueError("UHI data must contain 'UHI Index' column.")
    if 'datetime' not in uhi_data.columns: # Ensure datetime column exists
         raise ValueError("UHI data must contain a 'datetime' column.")

    x_coords = uhi_data['Longitude'].values
    y_coords = uhi_data['Latitude'].values
    x_grid = np.clip(np.floor((x_coords - topleft_lon) / x_res_deg), 0, sat_W - 1).astype(int)
    y_grid = np.clip(np.floor((topleft_lat - y_coords) / y_res_deg), 0, sat_H - 1).astype(int)
    uhi_data = uhi_data.assign(x_grid=x_grid, y_grid=y_grid)

    timestamp_col_name = 'datetime'
    grouped = uhi_data.groupby(timestamp_col_name)

    for timestamp, group in tqdm(grouped, desc="Precomputing UHI grids"):
        grid = np.full((sat_H, sat_W), np.nan, dtype=np.float32)
        mask = np.zeros((sat_H, sat_W), dtype=bool)
        y_indices = group['y_grid'].values
        x_indices = group['x_grid'].values
        uhi_values = group['UHI Index'].values
        grid[y_indices, x_indices] = uhi_values
        mask[y_indices, x_indices] = True
        target_grids[timestamp] = grid
        valid_masks[timestamp] = mask

    return target_grids, valid_masks


def normalize_lst(lst_tensor: Optional[np.ndarray], sat_H: int, sat_W: int) -> np.ndarray:
    """Normalizes LST tensor from Kelvin to [-1, 1]. Returns zeros if input is None or empty."""
    if lst_tensor is None or not np.any(lst_tensor):
        return np.zeros((1, sat_H, sat_W), dtype=np.float32)
    # Assume input lst_tensor is already (1, H, W) or (H, W)
    if lst_tensor.ndim == 2: lst_tensor = lst_tensor[np.newaxis, :, :]
    if lst_tensor.shape[1:] != (sat_H, sat_W):
         logging.warning(f"Input LST shape {lst_tensor.shape[1:]} does not match target {(sat_H, sat_W)}. Resizing needed before calling normalize.")
         # Handle resizing outside this function or add it here if preferred
         return np.zeros((1, sat_H, sat_W), dtype=np.float32) # Return zeros if shape mismatch

    lst_min_k, lst_max_k = 250.0, 330.0 # Kelvin range for normalization
    norm_01 = (lst_tensor - lst_min_k) / (lst_max_k - lst_min_k)
    norm_neg1_pos1 = (norm_01 * 2.0) - 1.0
    return np.clip(norm_neg1_pos1, -1.0, 1.0).astype(np.float32)


# --- Weather Utilities --- #

def normalize_min_max(value: float | np.ndarray, var_name: str) -> float | np.ndarray:
    """Normalizes a weather variable using pre-defined min/max values."""
    params = WEATHER_VARIABLES_INFO.get(var_name)
    if not params:
        return value # No normalization defined
    min_val, max_val = params['min'], params['max']
    if max_val == min_val:
        return np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
    norm_01 = (value - min_val) / (max_val - min_val)
    # Clip to ensure values stay within [0, 1] after normalization
    return np.clip(norm_01, 0.0, 1.0)


def get_closest_weather_data(timestamp: pd.Timestamp,
                             bronx_weather: pd.DataFrame,
                             manhattan_weather: pd.DataFrame,
                             enabled_weather_features: List[str]) -> Dict[str, Dict[str, float]]:
    # Finds closest weather data from Bronx and Manhattan stations for a given timestamp, considering only enabled features.
    # Ensure weather dataframes have datetime index or column named 'datetime'
    if not isinstance(bronx_weather.index, pd.DatetimeIndex) and 'datetime' not in bronx_weather.columns:
         raise ValueError("Bronx weather DataFrame must have a DatetimeIndex or a 'datetime' column.")
    if not isinstance(manhattan_weather.index, pd.DatetimeIndex) and 'datetime' not in manhattan_weather.columns:
         raise ValueError("Manhattan weather DataFrame must have a DatetimeIndex or a 'datetime' column.")

    # Use the correct column/index name
    dt_accessor_bronx = bronx_weather['datetime'] if 'datetime' in bronx_weather.columns else bronx_weather.index
    dt_accessor_manhattan = manhattan_weather['datetime'] if 'datetime' in manhattan_weather.columns else manhattan_weather.index

    bronx_idx = (dt_accessor_bronx - timestamp).abs().idxmin()
    manhattan_idx = (dt_accessor_manhattan - timestamp).abs().idxmin()

    bronx_data = bronx_weather.loc[bronx_idx]
    manhattan_data = manhattan_weather.loc[manhattan_idx]

    # weather_vars = ['air_temp', 'rel_humidity', 'avg_windspeed', 'wind_direction', 'solar_flux']
    # Use only enabled features that are base variables (wind_direction_sin/cos handled separately)
    relevant_vars_for_lookup = [f for f in enabled_weather_features if f in WEATHER_VARIABLES_INFO and f != 'wind_direction']
    if 'wind_direction' in enabled_weather_features:
        relevant_vars_for_lookup.append('wind_direction') # Ensure original wind_direction is fetched if selected

    return {
        'bronx': {var: bronx_data[var] for var in relevant_vars_for_lookup if var in bronx_data},
        'manhattan': {var: manhattan_data[var] for var in relevant_vars_for_lookup if var in manhattan_data}
    }

def build_weather_grid(timestamp: pd.Timestamp,
                       bronx_weather: pd.DataFrame,
                       manhattan_weather: pd.DataFrame,
                       bronx_coords: Tuple[float, float],
                       manhattan_coords: Tuple[float, float],
                       grid_coords: np.ndarray,
                       sat_H: int, sat_W: int,
                       enabled_weather_features: List[str]) -> np.ndarray:
    """Builds normalized weather grid using IDW interpolation for enabled features only."""
    
    actual_num_channels = calculate_actual_weather_channels(enabled_weather_features)
    if actual_num_channels == 0:
        logging.warning(f"No weather features enabled for timestamp {timestamp}. Returning empty grid of shape (0, {sat_H}, {sat_W})")
        return np.zeros((0, sat_H, sat_W), dtype=np.float32)

    station_data = get_closest_weather_data(timestamp, bronx_weather, manhattan_weather, enabled_weather_features)
    bronx_raw = station_data['bronx']
    manhattan_raw = station_data['manhattan']

    # Check if essential data for any enabled feature is missing
    # This is a basic check; more robust would be per-feature
    if not bronx_raw and any(f in enabled_weather_features for f in bronx_raw.keys()):
        logging.error(f"Missing Bronx weather data for enabled features at timestamp {timestamp}. Returning zeros for all channels.")
        return np.zeros((actual_num_channels, sat_H, sat_W), dtype=np.float32)
    if not manhattan_raw and any(f in enabled_weather_features for f in manhattan_raw.keys()):
        logging.error(f"Missing Manhattan weather data for enabled features at timestamp {timestamp}. Returning zeros for all channels.")
        return np.zeros((actual_num_channels, sat_H, sat_W), dtype=np.float32)

    # Initialize weather grid with the correct number of channels
    weather_grid = np.zeros((actual_num_channels, sat_H, sat_W), dtype=np.float32)
    
    grid_lat_raw = grid_coords[:, 0]
    grid_lon_raw = grid_coords[:, 1]
    lat_grid = grid_lat_raw.reshape(sat_H, sat_W)
    lon_grid = grid_lon_raw.reshape(sat_H, sat_W)

    dist_sq_bronx = (lat_grid - bronx_coords[0])**2 + (lon_grid - bronx_coords[1])**2
    dist_sq_manhattan = (lat_grid - manhattan_coords[0])**2 + (lon_grid - manhattan_coords[1])**2
    epsilon = 1e-9
    weight_bronx = 1.0 / (dist_sq_bronx + epsilon)
    weight_manhattan = 1.0 / (dist_sq_manhattan + epsilon)
    total_weight = weight_bronx + weight_manhattan
    # Handle cases where a grid cell is exactly at a station location (total_weight might be huge or inf)
    # or where total_weight is zero (e.g. if epsilon was too small and both dists were zero, though unlikely)
    valid_weights = total_weight > epsilon 
    norm_weight_bronx = np.zeros_like(total_weight)
    norm_weight_manhattan = np.zeros_like(total_weight)

    norm_weight_bronx[valid_weights] = weight_bronx[valid_weights] / total_weight[valid_weights]
    norm_weight_manhattan[valid_weights] = weight_manhattan[valid_weights] / total_weight[valid_weights]

    # Fill cells with invalid weights (e.g. directly on a station, or numerical issue)
    # Heuristic: assign to nearest station if total_weight is zero or problematic
    # If a point is exactly on Bronx station, weight_bronx is inf, weight_manhattan is finite -> norm_weight_bronx = 1
    # If a point is equidistant from both (and not on either), weights are equal.
    # This logic should generally handle points directly on stations correctly due to division by total_weight.
    # However, if a point is on Bronx, and dist_sq_manhattan is also ~0, total_weight could be inf/inf.
    # A simple robust way: if not valid_weights, check which distance is smaller.
    not_valid_weights_mask = ~valid_weights
    if np.any(not_valid_weights_mask):
        bronx_closer = dist_sq_bronx[not_valid_weights_mask] < dist_sq_manhattan[not_valid_weights_mask]
        manhattan_closer_or_equal = dist_sq_manhattan[not_valid_weights_mask] <= dist_sq_bronx[not_valid_weights_mask]
        
        norm_weight_bronx[not_valid_weights_mask] = np.where(bronx_closer, 1.0, 0.0)
        norm_weight_manhattan[not_valid_weights_mask] = np.where(manhattan_closer_or_equal, 1.0, 0.0)
        # If equidistant and problematic, this might assign 1.0 to Manhattan, adjust if specific behavior needed.

    current_channel_idx = 0
    for feature_name in CANONICAL_WEATHER_FEATURE_ORDER:
        if feature_name == 'wind_direction_sin' or feature_name == 'wind_direction_cos':
            if 'wind_direction' not in enabled_weather_features:
                continue # Skip sin/cos if base wind_direction is not enabled
            # Wind direction (handled as a pair)
            if 'wind_direction' not in bronx_raw or 'wind_direction' not in manhattan_raw:
                logging.warning(f"Base 'wind_direction' data missing for {timestamp} at one or both stations. Wind components will be zero.")
                # weather_grid[current_channel_idx] remains zero
                # weather_grid[current_channel_idx+1] remains zero
            else:
                wd_bronx_rad = np.deg2rad(bronx_raw['wind_direction'])
                wd_manhattan_rad = np.deg2rad(manhattan_raw['wind_direction'])
                sin_bronx, cos_bronx = np.sin(wd_bronx_rad), np.cos(wd_bronx_rad)
                sin_manhattan, cos_manhattan = np.sin(wd_manhattan_rad), np.cos(wd_manhattan_rad)
                interp_sin = sin_bronx * norm_weight_bronx + sin_manhattan * norm_weight_manhattan
                interp_cos = cos_bronx * norm_weight_bronx + cos_manhattan * norm_weight_manhattan
                length = np.sqrt(interp_sin**2 + interp_cos**2 + epsilon)
                valid_length = length > epsilon
                
                if feature_name == 'wind_direction_sin':
                    weather_grid[current_channel_idx][valid_length] = interp_sin[valid_length] / length[valid_length]
                elif feature_name == 'wind_direction_cos':
                    weather_grid[current_channel_idx][valid_length] = interp_cos[valid_length] / length[valid_length]
            current_channel_idx +=1

        elif feature_name in enabled_weather_features: # Handle other scalar features
            if feature_name not in bronx_raw or feature_name not in manhattan_raw:
                logging.warning(f"Weather variable '{feature_name}' missing for {timestamp} at one or both stations. Channel will be zero.")
                # weather_grid[current_channel_idx] remains zero
            else:
                # Use WEATHER_VARIABLES_INFO for normalization parameters
                norm_params = WEATHER_VARIABLES_INFO[feature_name]['norm_params']
                val_bronx_norm = (bronx_raw[feature_name] - norm_params['min']) / (norm_params['max'] - norm_params['min'] + epsilon)
                val_manhattan_norm = (manhattan_raw[feature_name] - norm_params['min']) / (norm_params['max'] - norm_params['min'] + epsilon)
                val_bronx_norm = np.clip(val_bronx_norm, 0.0, 1.0)
                val_manhattan_norm = np.clip(val_manhattan_norm, 0.0, 1.0)
                
                interpolated_norm_val = (val_bronx_norm * norm_weight_bronx + val_manhattan_norm * norm_weight_manhattan)
                weather_grid[current_channel_idx] = interpolated_norm_val
            current_channel_idx += 1
            
    return weather_grid

# --- Time/Metadata Utilities --- #

def get_time_embedding(timestamp: pd.Timestamp, sat_H: int, sat_W: int) -> np.ndarray:
    """Computes sin/cos embeddings for minute of day."""
    if not isinstance(timestamp, pd.Timestamp):
         raise TypeError(f"Input must be a pandas Timestamp, got {type(timestamp)}")
    total_minutes = timestamp.hour * 60 + timestamp.minute
    norm_minute = total_minutes / 1440.0
    minute_sin = np.sin(2 * np.pi * norm_minute)
    minute_cos = np.cos(2 * np.pi * norm_minute)
    time_features = np.array([minute_sin, minute_cos], dtype=np.float32)
    # Tile across spatial dimensions
    time_map = np.tile(time_features[:, np.newaxis, np.newaxis], (1, sat_H, sat_W))
    return time_map

def normalize_clay_timestamp(date: pd.Timestamp) -> np.ndarray:
    """Normalizes timestamp for Clay encoder (week + hour sin/cos)."""
    if not isinstance(date, pd.Timestamp):
         raise TypeError(f"Input must be a pandas Timestamp, got {type(date)}")
    week_of_year = date.isocalendar().week
    hour_of_day = date.hour
    norm_week = (week_of_year - 1) / 52.0
    norm_hour = hour_of_day / 24.0
    week_sin, week_cos = np.sin(2 * np.pi * norm_week), np.cos(2 * np.pi * norm_week)
    hour_sin, hour_cos = np.sin(2 * np.pi * norm_hour), np.cos(2 * np.pi * norm_hour)
    return np.array([week_sin, week_cos, hour_sin, hour_cos], dtype=np.float32)

def normalize_clay_latlon(bounds: List[float]) -> np.ndarray:
    """Calculates normalized sin/cos of dataset center lat/lon for Clay."""
    min_lon, min_lat, max_lon, max_lat = bounds
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    lat_rad, lon_rad = np.deg2rad(center_lat), np.deg2rad(center_lon)
    return np.array([math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)], dtype=np.float32)

# --- File Download/Unzip Utilities (from download_data.ipynb) --- #

def download_file(url: str, output_path: Path) -> bool:
    """Downloads a file from a URL to a given path with progress bar."""
    if not url:
        logging.error(f"No URL provided for {output_path.name}.")
        return False
    if output_path.exists():
        logging.info(f"File {output_path.name} already exists. Skipping download.")
        return True
    try:
        logging.info(f"Downloading {output_path.name} from {url}...")
        response = requests.get(url, stream=True, timeout=120) # Increased timeout
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        logging.info(f"Successfully downloaded {output_path.name}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {output_path.name}: {e}")
        if output_path.exists(): # Clean up partial download
            os.remove(output_path)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during download of {output_path.name}: {e}")
        if output_path.exists(): # Clean up partial download
             os.remove(output_path)
        return False

def unzip_file(zip_path: Path, extract_dir: Path) -> bool:
    """Unzips a file to a specified directory using the 'unzip' command-line tool."""
    if not zip_path.exists():
        logging.error(f"Zip file not found: {zip_path}")
        return False
    # Check if the final TIF already exists to avoid unnecessary unzipping
    expected_tif_name = zip_path.stem + ".tif"
    expected_tif_path = extract_dir / expected_tif_name
    if expected_tif_path.exists():
        logging.info(f"Expected TIF file {expected_tif_path.name} already exists. Skipping unzip.")
        return True

    try:
        logging.info(f"Unzipping {zip_path.name} to {extract_dir}...")
        # Use -o to overwrite existing files without prompting
        subprocess.run(['unzip', '-o', str(zip_path), '-d', str(extract_dir)],
                                capture_output=True, text=True, check=True, timeout=600) # Increased timeout
        logging.info(f"Successfully unzipped {zip_path.name}")

        # Verify the expected TIF file exists after unzipping
        if not expected_tif_path.exists():
             logging.warning(f"Expected TIF file {expected_tif_path.name} not found directly in {extract_dir} after unzipping.")
             # Look for any .tif file as a fallback check
             tif_files = list(extract_dir.glob('*.tif')) + list(extract_dir.glob('*/*.tif'))
             if tif_files:
                 logging.info(f"  Found other TIF files: {[f.name for f in tif_files]}. Manual check might be needed.")
             else:
                 logging.warning(f"  No .tif files found in {extract_dir} or immediate subdirectories.")
             return False # Consider it failed if the *specific* expected file isn't there
        return True
    except FileNotFoundError:
        logging.error("Error: 'unzip' command not found. Please install it.")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"Error: Unzipping {zip_path.name} timed out.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error unzipping {zip_path.name}: {e}")
        logging.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during unzipping of {zip_path.name}: {e}")
        return False

# --- Command Execution Helpers --- #

def run_command(command: List[str], description: str, timeout: int = 600):
    """Runs a subprocess command, logs output, and handles errors."""
    logging.info(f"Running: {description}...")
    logging.debug(f"Command: {' '.join(command)}")
    start_time = time.time()
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        logging.debug(f"STDOUT:\n{result.stdout}")
        logging.debug(f"STDERR:\n{result.stderr}")
        elapsed = time.time() - start_time
        logging.info(f"{description} completed successfully in {elapsed:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {description}:")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        logging.error(f"Error: Command '{' '.join(command)}' timed out after {timeout} seconds.")
        raise
    except FileNotFoundError:
        logging.error(f"Error: Command '{command[0]}' not found. Ensure required tools (e.g., GDAL, PDAL) are installed and in PATH.")
        raise

def reproject_raster(input_path: Path, output_path: Path, target_crs: str, resampling_method: str ='bilinear', nodata_val: Any = -9999):
    """Reprojects a raster using gdalwarp."""
    cmd = [
        'gdalwarp',
        '-t_srs', target_crs,
        '-r', resampling_method,
        '-overwrite', # Allow overwriting existing output
        '-dstnodata', str(nodata_val), # Explicitly set nodata for output
        str(input_path),
        str(output_path)
    ]
    run_command(cmd, f"Reprojecting {input_path.name} to {target_crs}")

# --- Resampling Utility --- #

def resample_xarray_to_target(
    data_xr: xr.DataArray,
    target_height: int,
    target_width: int,
    target_transform: Any, # Affine transform
    target_crs: Any, # CRS object or string
    resampling_method: Resampling = Resampling.bilinear,
    fill_value: Optional[float] = None
) -> Optional[np.ndarray]:
    """Resamples an xarray.DataArray to a target grid using rioxarray.
    
    Args:
        data_xr (xr.DataArray): Input data array with spatial coords and CRS.
        target_height (int): Target height in pixels.
        target_width (int): Target width in pixels.
        target_transform (Affine): Target affine transformation.
        target_crs: Target Coordinate Reference System (CRS object or string).
        resampling_method (Resampling): Resampling algorithm (default: bilinear).
        fill_value (Optional[float]): Value to fill nodata areas (default: None).
        
    Returns:
        Optional[np.ndarray]: Resampled array with shape [bands, height, width] or None if error.
    """
    try:
        # Check if the input data has multiple bands when it should have only one
        if len(data_xr.shape) >= 3:
            logging.debug(f"Input data has {data_xr.shape[0]} bands before resampling.")

        # Get bounds from transform and dimensions
        left, top = target_transform.c, target_transform.f
        right = left + target_transform.a * target_width
        bottom = top + target_transform.e * target_height  # e is negative

        # Make sure data_xr has a CRS set
        if not hasattr(data_xr, 'rio') or not hasattr(data_xr.rio, 'crs'):
            raise ValueError("Input xarray DataArray must have CRS information from rioxarray")

        resampled_xr = data_xr.rio.reproject(
            dst_crs=target_crs,
            shape=(target_height, target_width),
            transform=target_transform,
            resampling=resampling_method,
            # Use the source nodata if available, otherwise, it might infer
            nodata=data_xr.rio.nodata
        )

        resampled_np = resampled_xr.to_numpy()
        
        # Handle fill value for potential new nodata areas introduced by resampling
        if fill_value is not None:
            # Check if nodata exists and fill if necessary
            current_nodata = resampled_xr.rio.nodata
            if current_nodata is not None:
                 if np.isnan(current_nodata):
                     resampled_np[np.isnan(resampled_np)] = fill_value
            else:
                     resampled_np[resampled_np == current_nodata] = fill_value
            # Also fill NaNs that might not be explicitly marked as nodata
            resampled_np[np.isnan(resampled_np)] = fill_value

        # Ensure channel dimension exists if needed (e.g., for single-band data)
        if resampled_np.ndim == 2:
            resampled_np = resampled_np[np.newaxis, :, :]
        
        logging.debug(f"Resampled data shape (after conversion to numpy): {resampled_np.shape}")
        
        # Ensure NaN values are preserved during resampling
        if 'nodata' in resampled_xr.attrs:
            nodata_val = resampled_xr.attrs['nodata']
            # Replace nodata values with NaN for consistent handling downstream
            if not np.isnan(nodata_val):
                resampled_np = np.where(resampled_np == nodata_val, np.nan, resampled_np)
        
        return resampled_np.astype(np.float32)

    except Exception as e:
        logging.error(f"Error during resampling: {e}", exc_info=True)
        return None 