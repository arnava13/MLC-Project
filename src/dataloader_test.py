import torch
from torch.utils.data import DataLoader
from ingest.dataloaders import CityDataSet  

dataset = CityDataSet(
    bounds=(-74.01, 40.75, -73.86, 40.88),  
    averaging_window=7,
    selected_bands=["B02", "B03", "B04", "B08"],
    resolution_m=10,
    uhi_csv="./src/ingest/sample_data/sample_uhi.csv", 
    bbox_csv="./src/ingest/sample_data/sample_bbox.csv"
)

# testing the dataset
print(len(dataset))
x, y = dataset[0]
print(x.shape) 
print(y) 
