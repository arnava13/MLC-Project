import torch
from torch.utils.data import DataLoader
from ingest.dataloaders import CityDataSet  

# Instantiate the dataset
dataset = CityDataSet(
    bounds=(-74.01, 40.75, -73.86, 40.88),  # Bounding box (NYC region)
    averaging_window=7,
    selected_bands=["B02", "B03", "B04", "B08"],  # Sentinel-2 bands
    resolution_m=10,
    uhi_csv="./src/ingest/sample_data/sample_uhi.csv", 
    bbox_csv="./src/ingest/sample_data/sample_bbox.csv",
    weather_csv="./src/ingest/sample_data/nyc_weather_grid.csv",
    include_lst=False  # Set to True to include Landsat-based LST
)

# Test one sample
print(len(dataset))  # Number of UHI observations
x, weather, meta = dataset[0]
print(x.shape)       # (4, H, W) or (5, H, W) if LST included
print(weather)       # [temp_max, temp_min, precip]
print(meta.shape)    # (5,) → [x_grid, y_grid, min_since_midnight, month, UHI]
