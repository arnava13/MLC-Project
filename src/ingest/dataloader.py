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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CityDataSet(Dataset):
    """
    PyTorch Dataset for UHI modeling using locally stored data.
    Loads a cloudless mosaic & optionally a single LST median for static features,
    plus dynamic weather/time data. Returns a dictionary for the model.
    """

    def __init__(self, bounds, averaging_window,
                 resolution_m,
                 uhi_csv, bbox_csv, weather_csv, cloudless_mosaic_path,
                 data_dir, city_name,
                 include_lst=True,
                 single_lst_median_path: str = None): # Added path for single LST
        """
        Initialize the dataset.

        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]
            averaging_window: Days lookback for LST median (used only if single_lst_median_path not provided)
            resolution_m: Target spatial resolution (meters) for UHI/Weather/LST grids.
            uhi_csv: Path to UHI data CSV file.
            bbox_csv: Path to bounding box CSV file.
            weather_csv: Path to weather data CSV file.
            cloudless_mosaic_path: Path to the pre-generated cloudless Sentinel-2 mosaic (.npy).
            data_dir: Base directory for stored data.
            city_name: Name of the city.
            include_lst: Whether to include Land Surface Temperature data.
            single_lst_median_path (str, optional): Path to a pre-generated single LST median .npy file.
                                                    If provided, ignores averaging_window and dynamic loading.
        """
        # --- Basic Parameters ---
        self.bounds = bounds
        self.include_lst = include_lst
        self.resolution_m = resolution_m
        self.bbox_csv = Path(bbox_csv)
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
        self.timestamps = pd.to_datetime(self.uhi_data['timestamp'])
        self.sat_H, self.sat_W = self._determine_target_grid_size()

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
             self.single_lst_median = np.zeros((1, 1, self.sat_H, self.sat_W), dtype=np.float32)

        # --- Precompute UHI Grids/Masks & Load Weather ---
        self.target_grids, self.valid_masks = self._precompute_uhi_grids()
        self.weather_df = pd.read_csv(weather_csv)
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        self._prepare_weather_metadata()

        logging.info(f"Dataset initialized for {self.city_name} with {len(self)} samples. LST included: {self.include_lst}")
        logging.info(f"Target grid size (H, W): ({self.sat_H}, {self.sat_W})")

    def _determine_target_grid_size(self):
        """Determine the target grid size (H, W) based on bbox and resolution_m."""
        if not self.bbox_csv.exists():
             raise FileNotFoundError(f"Bounding box CSV not found: {self.bbox_csv}")
        bbox_data = pd.read_csv(self.bbox_csv)
        min_lat, max_lat = bbox_data['latitudes'].min(), bbox_data['latitudes'].max()
        min_lon, max_lon = bbox_data['longitudes'].min(), bbox_data['longitudes'].max()

        deg_per_meter_lat = 1 / 111000
        deg_per_meter_lon = 1 / (111320 * math.cos(math.radians((min_lat + max_lat) / 2)))
        height_deg = max_lat - min_lat
        width_deg = max_lon - min_lon

        H = math.ceil(height_deg / (self.resolution_m * deg_per_meter_lat))
        W = math.ceil(width_deg / (self.resolution_m * deg_per_meter_lon))
        return max(1, H), max(1, W)

    def _precompute_uhi_grids(self):
        """Create target UHI grids and masks for all timestamps at target resolution."""
        target_grids = []
        valid_masks = []
        bbox_data = pd.read_csv(self.bbox_csv)
        topleft_lat = bbox_data['latitudes'].max()
        topleft_lon = bbox_data['longitudes'].min()
        deg_per_meter_lat = 1 / 111000
        deg_per_meter_lon = 1 / (111320 * math.cos(math.radians((bbox_data['latitudes'].min() + bbox_data['latitudes'].max()) / 2)))

        x_res_deg = self.resolution_m * deg_per_meter_lon
        y_res_deg = self.resolution_m * deg_per_meter_lat

        self.uhi_data['x_grid'] = np.clip(np.floor((self.uhi_data['longitudes'] - topleft_lon) / x_res_deg), 0, self.sat_W - 1).astype(int)
        self.uhi_data['y_grid'] = np.clip(np.floor((topleft_lat - self.uhi_data['latitudes']) / y_res_deg), 0, self.sat_H - 1).astype(int)

        grouped = self.uhi_data.groupby(self.timestamps)
        for timestamp, group in tqdm(grouped, desc="Precomputing UHI grids"):
            grid = np.full((self.sat_H, self.sat_W), np.nan, dtype=np.float32)
            mask = np.zeros((self.sat_H, self.sat_W), dtype=bool)
            y_indices = group['y_grid'].values
            x_indices = group['x_grid'].values
            uhi_values = group['UHI'].values
            grid[y_indices, x_indices] = uhi_values
            mask[y_indices, x_indices] = True
            target_grids.append(grid)
            valid_masks.append(mask)
        return target_grids, valid_masks

    def _prepare_weather_metadata(self):
        """Pre-compute metadata for weather grid construction."""
        self.weather_lat_vals = np.sort(self.weather_df['lat'].unique())
        self.weather_lon_vals = np.sort(self.weather_df['lon'].unique())
        if len(self.weather_lat_vals) < 2 or len(self.weather_lon_vals) < 2:
            raise ValueError("Weather CSV must contain at least a 2x2 grid of lat/lon points.")
        self.weather_lat_step = np.abs(np.diff(self.weather_lat_vals)).mean() if len(self.weather_lat_vals) > 1 else 1.0
        self.weather_lon_step = np.abs(np.diff(self.weather_lon_vals)).mean() if len(self.weather_lon_vals) > 1 else 1.0
        self.weather_H_orig = len(self.weather_lat_vals)
        self.weather_W_orig = len(self.weather_lon_vals)

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

    def _build_weather_grid(self, date: pd.Timestamp) -> np.ndarray:
        """Return weather grid (C_w=3, H, W) for given date, resized to target H, W."""
        rows = self.weather_df[self.weather_df['date'] == date.normalize()]
        grid = np.full((3, self.weather_H_orig, self.weather_W_orig), np.nan, dtype=np.float32)
        weather_lat_min = self.weather_lat_vals[0]
        weather_lon_min = self.weather_lon_vals[0]
        lat_indices = np.round((rows['lat'].values - weather_lat_min) / self.weather_lat_step).astype(int)
        lon_indices = np.round((rows['lon'].values - weather_lon_min) / self.weather_lon_step).astype(int)
        valid_idx = (0 <= lat_indices) & (lat_indices < self.weather_H_orig) & \
                    (0 <= lon_indices) & (lon_indices < self.weather_W_orig)
        lat_indices = lat_indices[valid_idx]
        lon_indices = lon_indices[valid_idx]
        valid_rows = rows.iloc[valid_idx]
        grid[0, lat_indices, lon_indices] = valid_rows['temp_max'].values
        grid[1, lat_indices, lon_indices] = valid_rows['temp_min'].values
        grid[2, lat_indices, lon_indices] = valid_rows['precip'].values
        grid = np.nan_to_num(grid)
        if (self.weather_H_orig, self.weather_W_orig) != (self.sat_H, self.sat_W):
            zoom_factors = (1, self.sat_H / self.weather_H_orig, self.sat_W / self.weather_W_orig)
            grid = zoom(grid, zoom=zoom_factors, order=1)
        return grid

    def _get_time_embedding(self, timestamp: pd.Timestamp) -> np.ndarray:
        """Computes sin/cos embeddings for minute of day and day of year."""
        total_minutes = timestamp.hour * 60 + timestamp.minute
        day_of_year = timestamp.dayofyear
        norm_minute = total_minutes / 1440.0
        norm_day = (day_of_year - 1) / 365.0
        minute_sin = np.sin(2 * np.pi * norm_minute)
        minute_cos = np.cos(2 * np.pi * norm_minute)
        day_sin = np.sin(2 * np.pi * norm_day)
        day_cos = np.cos(2 * np.pi * norm_day)
        time_features = np.array([minute_sin, minute_cos, day_sin, day_cos], dtype=np.float32)
        time_map = np.tile(time_features[:, np.newaxis, np.newaxis], (1, self.sat_H, self.sat_W))
        return time_map

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        timestamp = self.timestamps[idx]

        weather_grid = self._build_weather_grid(timestamp)
        weather_seq = weather_grid[np.newaxis, ...]

        time_map = self._get_time_embedding(timestamp)
        time_map_seq = time_map[np.newaxis, ...]

        # Get the static LST median (already normalized or zero)
        # It will have shape (1, H, W)
        lst_tensor = self.single_lst_median if self.include_lst else np.zeros((1, self.sat_H, self.sat_W), dtype=np.float32)
        # Add sequence dimension T=1 -> (1, 1, H, W)
        lst_seq = lst_tensor[np.newaxis, ...]

        target = self.target_grids[idx]
        mask = self.valid_masks[idx].astype(np.float32)
        target = np.nan_to_num(target)

        cloudless_mosaic = self.cloudless_mosaic.astype(np.float32)

        return {
            "cloudless_mosaic": cloudless_mosaic,
            "weather_seq": weather_seq.astype(np.float32),
            "time_emb_seq": time_map_seq.astype(np.float32),
            "lst_seq": lst_seq.astype(np.float32), # Now always contains the single median (or zeros)
            "target": target.astype(np.float32),
            "mask": mask.astype(np.float32)
        }

# TODO: Verify selected_bands is no longer needed if Clay uses fixed bands
# TODO: Ensure cloudless_mosaic resolution aligns with how Clay expects input


# TODO: Create dataloaders for the models in models.py. 