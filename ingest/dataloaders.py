import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ingest.load_satellite_data import load_lst_tensor_from_bbox, load_sentinel_tensor_from_bbox
from scipy.ndimage import zoom
from geopy.distance import geodesic

class CityDataSet(Dataset):
    def __init__(self, bounds, averaging_window, selected_bands, resolution_m, include_lst=True, uhi_csv=None, bbox_csv=None):
        # Parameters
        self.bounds = bounds
        self.averaging_window = averaging_window # Number of days before UHI Observation Day
        self.selected_bands = selected_bands
        self.include_lst = include_lst
        self.resolution_m = resolution_m

        # Data
        self.uhi_csv = uhi_csv
        self.bbox_csv = bbox_csv
        self.uhi_data = self.load_uhi_data()
        self.satellite_tensor = self.load_satellite_tensor()



    def load_uhi_data(self):
        uhi_data = pd.read_csv(self.uhi_csv)
        timestamps = pd.to_datetime(uhi_data['timestamp'])
        latitudes = uhi_data['latitudes']
        longitudes = uhi_data['longitudes']
        bbox_data = pd.read_csv(self.bbox_csv) # e.g [-76.7604, 39.1905,-76.4274, 39.429]

        # Create a resolution_m grid from the latitudes and longitudes
        topleft_lat = bbox_data['latitudes'].iloc[0]
        topleft_lon = bbox_data['longitudes'].iloc[0]
        bottomright_lat = bbox_data['latitudes'].iloc[1]
        bottomright_lon = bbox_data['longitudes'].iloc[1]

        # Get corner differences in decimal degrees, convert to metres then convert to grid dimensions
        lat_diff = topleft_lat - bottomright_lat
        lon_diff = topleft_lon - bottomright_lon

        vertical_size_metres = geodesic((topleft_lat, topleft_lon), (bottomright_lat, topleft_lon)).meters
        horizontal_size_metres = geodesic((topleft_lat, topleft_lon), (topleft_lat, bottomright_lon)).meters

        x_size = horizontal_size_metres / self.resolution_m
        y_size = vertical_size_metres / self.resolution_m

        # Find the grid cell that each UHI observation falls into
        uhi_data['x_grid'] = np.floor((uhi_data['longitudes'] - topleft_lon) / self.resolution_m)
        uhi_data['y_grid'] = np.floor((uhi_data['latitudes'] - topleft_lat) / self.resolution_m)


        # Now get the timestamp as minutes since midnight and month of year (use in sinusoidal positional encodings)
        uhi_data['min_since_midnight'] = (uhi_data['timestamp'].dt.hour * 60 + uhi_data['timestamp'].dt.minute)
        uhi_data['month'] = uhi_data['timestamp'].dt.month


        # drop unused columns
        uhi_data = uhi_data[['x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']]

        # convert to numpy array with columns x_grid, y_grid, min_since_midnight, month, UHI
        uhi_data = uhi_data.to_numpy()

        return uhi_data

    def load_satellite_tensor(self, idx):
        # Get the time window for the satellite data
        time_window = self.uhi_data['timestamp'].iloc[idx] - self.averaging_window/24, self.uhi_data['timestamp'].iloc[idx] + self.averaging_window/24
        optical_tensors = load_sentinel_tensor_from_bbox(self.bounds, self.time_window, self.selected_bands)

        if self.include_lst:
            lst_tensors = load_lst_tensor_from_bbox(self.bounds, self.time_window)
            _, h_s, w_s = optical_tensors.shape
            _, h_l, w_l = lst_tensors.shape
            zoom_factors = (h_s / h_l, w_s / w_l)

            print(f"Resizing LST: zoom={zoom_factors}")
            lst_resized = zoom(lst_tensors[0], zoom_factors, order=1)  # Linear interpolation
            lst_resized = lst_resized[np.newaxis, :, :]  # Shape=(1, H, W)

        # Combine optical and LST tensors
        combined = np.concatenate([optical_tensors, lst_resized], axis=0)
        print("Combined tensor shape:", combined.shape)
        return combined
    
    def __len__(self):
        return len(self.satellite_tensors)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.satellite_tensors[idx])
    
    
    
    
