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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CityDataSet(Dataset):
    """
    PyTorch Dataset for UHI modeling using locally stored satellite data.
    Returns (satellite_tensor, weather_tensor, meta_tensor) for each UHI observation.
    """

    def __init__(self, bounds, averaging_window, selected_bands, resolution_m,
                 include_lst=True, uhi_csv=None, bbox_csv=None, weather_csv=None,
                 data_dir=None, city_name=None):
        """
        Initialize the dataset with local satellite data files.
        
        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]
            averaging_window: Number of days to look back for median composites
            selected_bands: List of Sentinel-2 bands used
            resolution_m: Spatial resolution in meters
            include_lst: Whether to include Land Surface Temperature data
            uhi_csv: Path to UHI data CSV file
            bbox_csv: Path to bounding box CSV file
            weather_csv: Path to weather data CSV file
            data_dir: Base directory for stored satellite data
            city_name: Name of the city (used for directory paths)
        """
        # Set basic parameters
        self.bounds = bounds
        self.averaging_window = averaging_window
        self.selected_bands = selected_bands
        self.include_lst = include_lst
        self.resolution_m = resolution_m
        self.bbox_csv = bbox_csv
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.city_name = city_name if city_name else "unknown_city"
        self.sat_files_dir = self.data_dir / self.city_name / "sat_files"
        
        # Check if satellite data directory exists
        if not self.sat_files_dir.exists():
            raise ValueError(f"Satellite data directory not found: {self.sat_files_dir}")
        
        # Load lookup table for satellite data
        self.lookup_path = self.sat_files_dir / "timewindow_lookup.json"
        if not self.lookup_path.exists():
            raise ValueError(f"Timewindow lookup file not found: {self.lookup_path}")
            
        with open(self.lookup_path, 'r') as f:
            self.lookup_table = json.load(f)

        # Load UHI CSV and parse timestamp
        self.uhi_data = pd.read_csv(uhi_csv)
        self.timestamps = pd.to_datetime(self.uhi_data['timestamp'])
        self.load_uhi_data()

        # Load weather CSV (daily max/min temp + precipitation)
        self.weather_df = pd.read_csv(weather_csv)
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        
        # Preload satellite (and optional LST) tensors
        self.satellite_tensors = self.load_satellite_tensor_from_files()

    def load_uhi_data(self):
        """Load UHI data and compute grid coordinates."""
        # Compute grid index and time features from lat/lon and timestamp
        bbox_data = pd.read_csv(self.bbox_csv)
        topleft_lat = bbox_data['latitudes'].iloc[0]
        topleft_lon = bbox_data['longitudes'].iloc[0]
        deg_per_meter = 1 / 111000
        x_res_deg = self.resolution_m * deg_per_meter
        y_res_deg = self.resolution_m * deg_per_meter

        self.uhi_data['x_grid'] = np.floor((self.uhi_data['longitudes'] - topleft_lon) / x_res_deg)
        self.uhi_data['y_grid'] = np.floor((self.uhi_data['latitudes'] - topleft_lat) / y_res_deg)
        self.uhi_data['min_since_midnight'] = self.timestamps.dt.hour * 60 + self.timestamps.dt.minute
        self.uhi_data['month'] = self.timestamps.dt.month

        # Keep only necessary columns (including lat/lon/timestamp for weather matching)
        self.uhi_data = self.uhi_data[['latitudes', 'longitudes', 'timestamp',
                                       'x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']]

    def load_satellite_tensor_from_files(self):
        """Load satellite tensors from local files."""
        all_tensors = []
        missing_data_count = 0
        
        for timestamp in self.timestamps:
            start_date = (timestamp - pd.Timedelta(days=self.averaging_window)).strftime("%Y-%m-%d")
            end_date = timestamp.strftime("%Y-%m-%d")
            time_window = f"{start_date}/{end_date}"
            
            # Check if we have data for this time window
            if time_window not in self.lookup_table:
                logging.warning(f"No data found for time window: {time_window}. Using zeros.")
                dummy = np.zeros((len(self.selected_bands) + (1 if self.include_lst else 0), 1, 1), dtype=np.float32)
                all_tensors.append(dummy)
                missing_data_count += 1
                continue
                
            try:
                # Get filenames from lookup table
                sentinel_filename = self.lookup_table[time_window]["sentinel"]
                sentinel_path = self.sat_files_dir / sentinel_filename
                
                # Load Sentinel data
                sentinel_tensor = np.load(sentinel_path)
                
                if self.include_lst:
                    lst_filename = self.lookup_table[time_window]["lst"]
                    if lst_filename:
                        lst_path = self.sat_files_dir / lst_filename
                        lst_tensor = np.load(lst_path)
                        
                        # Resize LST to match Sentinel tensor shape if needed
                        if lst_tensor.shape[1:] != sentinel_tensor.shape[1:]:
                            zoom_factors = (
                                1,  # channel dimension
                                sentinel_tensor.shape[1] / lst_tensor.shape[1],  # H direction
                                sentinel_tensor.shape[2] / lst_tensor.shape[2],  # W direction
                            )
                            lst_tensor = zoom(lst_tensor, zoom=zoom_factors, order=1)
                            
                        combined = np.concatenate([sentinel_tensor, lst_tensor], axis=0)
                    else:
                        # LST data not available for this time window
                        logging.warning(f"LST data not available for {time_window}. Using only Sentinel data.")
                        combined = sentinel_tensor
                else:
                    combined = sentinel_tensor
                    
                all_tensors.append(combined)
                
            except Exception as e:
                logging.error(f"Error loading satellite data for {time_window}: {e}")
                # On failure, append dummy tensor
                dummy = np.zeros((len(self.selected_bands) + (1 if self.include_lst else 0), 1, 1), dtype=np.float32)
                all_tensors.append(dummy)
                missing_data_count += 1
        
        if missing_data_count > 0:
            logging.warning(f"Missing satellite data for {missing_data_count} out of {len(self.timestamps)} timestamps.")
            
        return all_tensors

    def get_weather_for(self, lat, lon, timestamp):
        """Retrieve weather info for the nearest grid point and date."""
        # Retrieve weather info (max/min temp, precip) for the nearest grid point and date
        date = pd.to_datetime(timestamp).normalize()
        tolerance = 0.005  # ~500m tolerance for lat/lon (adjustable)

        match = self.weather_df[
            (np.abs(self.weather_df['lat'] - lat) <= tolerance) &
            (np.abs(self.weather_df['lon'] - lon) <= tolerance) &
            (self.weather_df['date'] == date)
        ]

        if len(match) == 0:
            logging.warning(f"No weather match for lat={lat}, lon={lon}, date={date.date()}")
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        return match[['temp_max', 'temp_min', 'precip']].iloc[0].values.astype(np.float32)

    def __len__(self):
        """Number of UHI samples."""
        return len(self.satellite_tensors)

    def __getitem__(self, idx):
        """Return (satellite tensor, weather info, meta features) for sample at index."""
        satellite = self.satellite_tensors[idx]
        uhi_row = self.uhi_data.iloc[idx]

        lat = uhi_row['latitudes']
        lon = uhi_row['longitudes']
        timestamp = pd.to_datetime(uhi_row['timestamp'])

        weather = self.get_weather_for(lat, lon, timestamp)
        meta = uhi_row[['x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']].to_numpy(dtype=np.float32)
        return satellite, weather, meta 
    

# -----------------------------------------------------------------------------
# 4. COLLATE FUNCTION FOR THE EXISTING DATALOADER -----------------------------
# -----------------------------------------------------------------------------

def uhi_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], T: int = 12):
    """Turn the current Dataset output into batched tensors compatible with UHINet.

    Each element of *batch* is (satellite, weather, meta) for *one* timestamp.
    We first group by sample index // T, assuming the DataLoader is ordered by
    timestamp per city/day.  For production you might build a proper Sequence
    Dataset, but this keeps the user’s existing CSV‑driven Dataset unchanged.
    """
    satellites, weathers, metas = zip(*batch)  # tuples of length B*T
    satellites = torch.stack([torch.from_numpy(s).float() for s in satellites])
    C, H, W = satellites.shape[1:]
    satellites = satellites.view(-1, T, C, H, W)  # (B, T, C, H, W)

    weathers = torch.from_numpy(np.stack(weathers)).float().view(-1, T, 3)
    metas = torch.from_numpy(np.stack(metas)).float().view(-1, T, metas[0].shape[-1])

    # Extract sin‑cos hour from meta[:, :, 2]
    minutes = metas[:, :, 2]
    phase = minutes / 1440 * 2 * math.pi
    time_feats = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # (B,T,2)
    return satellites, weathers, time_feats, metas[:, :, -1]  # last column is UHI target
