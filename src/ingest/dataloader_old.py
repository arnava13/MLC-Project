import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ingest.get_median import load_sentinel_tensor_from_bbox_median, load_lst_tensor_from_bbox_median
from scipy.ndimage import zoom

class CityDataSet(Dataset):
    """
    PyTorch Dataset for UHI modeling. For testing this dataloader, use dataloader_test.py
    Returns (satellite_tensor, weather_tensor, meta_tensor) for each UHI observation.
    """

    def __init__(self, bounds, averaging_window, selected_bands, resolution_m,
                 include_lst=True, uhi_csv=None, bbox_csv=None, weather_csv=None):
        # Set basic parameters
        self.bounds = bounds
        self.averaging_window = averaging_window
        self.selected_bands = selected_bands
        self.include_lst = include_lst
        self.resolution_m = resolution_m
        self.bbox_csv = bbox_csv

        # Load UHI CSV and parse timestamp
        self.uhi_data = pd.read_csv(uhi_csv)
        self.latitudes = self.uhi_data['latitudes']
        self.longitudes = self.uhi_data['longitudes']
        self.timestamps = pd.to_datetime(self.uhi_data['timestamp'])
        self.load_uhi_data()

        # Load weather CSV (daily max/min temp + precipitation)
        self.weather_df = pd.read_csv(weather_csv)
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])

        # Preload satellite (and optional LST) tensors
        self.satellite_tensors = self.load_satellite_tensor()

    def load_uhi_data(self):
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

    def load_satellite_tensor(self):
        # Load median composite Sentinel-2 (and optional LST) tensor for each timestamp
        all_tensors = []
        for timestamp in self.timestamps:
            start_date = (timestamp - pd.Timedelta(days=self.averaging_window)).strftime("%Y-%m-%d")
            end_date = timestamp.strftime("%Y-%m-%d")
            time_window = f"{start_date}/{end_date}"

            try:
                sentinel_tensor = load_sentinel_tensor_from_bbox_median(
                    bounds=self.bounds,
                    time_window=time_window,
                    selected_bands=self.selected_bands,
                    resolution_m=self.resolution_m
                )

                if self.include_lst:
                    # Load and resize LST to match Sentinel tensor shape
                    lst_tensor = load_lst_tensor_from_bbox_median(
                        bounds=self.bounds,
                        time_window=time_window,
                        resolution_m=30
                    )
                    zoom_factors = (
                        1,
                        sentinel_tensor.shape[1] / lst_tensor.shape[1],
                        sentinel_tensor.shape[2] / lst_tensor.shape[2]
                    )
                    lst_resized = zoom(lst_tensor, zoom=zoom_factors, order=1)
                    combined = np.concatenate([sentinel_tensor, lst_resized], axis=0)
                else:
                    combined = sentinel_tensor

                all_tensors.append(combined)

            except Exception as e:
                # On failure, append dummy tensor
                print(f"⚠️ Warning: Failed to load for {time_window} → {e}")
                dummy = np.zeros((len(self.selected_bands) + (1 if self.include_lst else 0), 1, 1), dtype=np.float32)
                all_tensors.append(dummy)

        return all_tensors

    def get_weather_for(self, lat, lon, timestamp):
        # Retrieve weather info (max/min temp, precip) for the nearest grid point and date
        date = pd.to_datetime(timestamp).normalize()
        tolerance = 0.005  # ~500m tolerance for lat/lon (adjustable)

        match = self.weather_df[
            (np.abs(self.weather_df['lat'] - lat) <= tolerance) &
            (np.abs(self.weather_df['lon'] - lon) <= tolerance) &
            (self.weather_df['date'] == date)
        ]

        if len(match) == 0:
            print(f"⚠️ No match for lat={lat}, lon={lon}, date={date.date()}")
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        return match[['temp_max', 'temp_min', 'precip']].iloc[0].values.astype(np.float32)

    def __len__(self):
        # Number of UHI samples
        return len(self.satellite_tensors)

    def __getitem__(self, idx):
        # Return (satellite tensor, weather info, meta features) for sample at index
        satellite = self.satellite_tensors[idx]
        uhi_row = self.uhi_data.iloc[idx]

        lat = uhi_row['latitudes']
        lon = uhi_row['longitudes']
        timestamp = pd.to_datetime(uhi_row['timestamp'])

        weather = self.get_weather_for(lat, lon, timestamp)
        meta = uhi_row[['x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']].to_numpy(dtype=np.float32)
        return satellite, weather, meta
