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

        # UHI (extract timestamps)
        self.uhi_data = pd.read_csv(uhi_csv)
        self.latitudes = self.uhi_data['latitudes']
        self.longitudes = self.uhi_data['longitudes']
        self.timestamps = pd.to_datetime(self.uhi_data['timestamp'])
        self.load_uhi_data()


        self.satellite_tensor = self.load_satellite_tensor()


        self.bbox_csv = bbox_csv


    def load_uhi_data(self):

        bbox_data = pd.read_csv(self.bbox_csv) # e.g [-76.7604, 39.1905,-76.4274, 39.429]

        # Create a resolution_m grid from the latitudes and longitudes
        topleft_lat = bbox_data['latitudes'].iloc[0]
        topleft_lon = bbox_data['longitudes'].iloc[0]
        bottomright_lat = bbox_data['latitudes'].iloc[1]
        bottomright_lon = bbox_data['longitudes'].iloc[1]

        # Find the grid cell that each UHI observation falls into
        self.uhi_data['x_grid'] = np.floor((self.uhi_data['longitudes'] - topleft_lon) / self.resolution_m)
        self.uhi_data['y_grid'] = np.floor((self.uhi_data['latitudes'] - topleft_lat) / self.resolution_m)


        # Now get the timestamp as minutes since midnight and month of year (use in sinusoidal positional encodings)
        self.uhi_data['min_since_midnight'] = (self.timestamps.dt.hour * 60 + self.timestamps.dt.minute)
        self.uhi_data['month'] = self.timestamps.dt.month


        # drop unused columns
        self.uhi_data = self.uhi_data[['x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']]

        # convert to numpy array with columns x_grid, y_grid, min_since_midnight, month, UHI
        self.uhi_data = self.uhi_data.to_numpy()

    def load_satellite_tensor(self, idx):
       
        return combined
    
    def __len__(self):
        return len(self.satellite_tensors)
    
    def __getitem__(self, idx):
        return self.satellite_tensors[idx], self.uhi_data[idx], self.weather_data[idx], self.lst_data[idx]

    
    
    
    
