import torch
from torch.utils.data import DataLoader
from ingest.dataloader import CityDataSet  

# Instantiate the dataset
dataset = CityDataSet(
    bounds=(-74.01, 40.75, -73.86, 40.88),  # Bounding box (NYC region)
    averaging_window=7,
    selected_bands=["B02", "B03", "B04", "B08"],  # Sentinel-2 bands
    resolution_m=10,
    uhi_csv="ingest/sample_data/sample_uhi.csv", 
    bbox_csv="ingest/sample_data/sample_bbox.csv",
    weather_csv="ingest/sample_data/nyc_weather_grid.csv",
    data_dir="ingest/sample_data",
    city_name="unknown_city",
    include_lst=True  # Include Landsat-based LST
)

# Test one sample
print(f"Dataset length: {len(dataset)}")  # Number of UHI observations
x, weather, meta = dataset[0]
print(f"Satellite tensor shape: {x.shape}")       # (4, H, W) or (5, H, W) if LST included
print(f"Weather data: {weather}")       # [temp_max, temp_min, precip]
print(f"Meta data shape: {meta.shape}")  # [x_grid, y_grid, min_since_midnight, month, UHI]
