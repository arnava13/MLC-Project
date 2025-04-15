import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# from ingest.load_satellite_data import load_lst_tensor_from_bbox, load_sentinel_tensor_from_bbox
from ingest.get_median import load_sentinel_tensor_from_bbox_median
from scipy.ndimage import zoom
# from geopy.distance import geodesic

class CityDataSet(Dataset):
    def __init__(self, bounds, averaging_window, selected_bands, resolution_m, include_lst=True, uhi_csv=None, bbox_csv=None):
        # Parameters
        self.bounds = bounds
        self.averaging_window = averaging_window # Number of days before UHI Observation Day
        self.selected_bands = selected_bands
        self.include_lst = include_lst
        self.resolution_m = resolution_m
        self.bbox_csv = bbox_csv

        # UHI (extract timestamps)
        self.uhi_data = pd.read_csv(uhi_csv)
        self.latitudes = self.uhi_data['latitudes']
        self.longitudes = self.uhi_data['longitudes']
        self.timestamps = pd.to_datetime(self.uhi_data['timestamp'])
        self.load_uhi_data()

        self.satellite_tensors = self.load_satellite_tensor()


    def load_uhi_data(self):
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

        self.uhi_data = self.uhi_data[['x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']]
        self.uhi_data = self.uhi_data.to_numpy()

    def load_satellite_tensor(self):
        all_tensors = []
        for timestamp in self.timestamps:
            start_date = (timestamp - pd.Timedelta(days=self.averaging_window)).strftime("%Y-%m-%d")
            end_date = timestamp.strftime("%Y-%m-%d")
            time_window = f"{start_date}/{end_date}"

            try:
                median_tensor = load_sentinel_tensor_from_bbox_median(
                    bounds=self.bounds,
                    time_window=time_window,
                    selected_bands=self.selected_bands,
                    resolution_m=self.resolution_m
                )
                all_tensors.append(median_tensor)
            except Exception as e:
                print(f"Warning: Failed to load for {time_window} → {e}")
                dummy = np.zeros((len(self.selected_bands), 1, 1), dtype=np.float32)
                all_tensors.append(dummy)

        return all_tensors

    def __len__(self):
        return len(self.satellite_tensors)
    
    def __getitem__(self, idx):
        return self.satellite_tensors[idx], self.uhi_data[idx]
