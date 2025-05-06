# Urban Heat Island (UHI) Prediction Project

This project predicts Urban Heat Island (UHI) effects using machine learning with remote sensing and weather data.

## Project Structure

- **src/**: Source code
  - **ingest/**: Data loading and preprocessing
    - `dataloader_branched.py`: Main data loading class with multi-input branch support
    - `data_utils.py`: Utilities for data processing including resampling
  - **Clay/**: Clay feature encoder for satellite imagery
  - **models/**: Neural network model definitions
- **notebooks/**: Jupyter notebooks
  - `download_data.ipynb`: Downloads and processes satellite, elevation, and weather data
- **data/**: Data storage (not versioned)

## Data Overview

The model uses the following data sources:
- Sentinel-2 multispectral imagery
- Digital Elevation Models (DEM) from USGS 3DEP
- Digital Surface Models (DSM) from USGS 3DEP
- Weather station data
- Land Surface Temperature (LST)

## Known Issues

- **Multi-band elevation data**: The DEM/DSM data from Planetary Computer's 3DEP collection sometimes returns multiple bands (5 bands) when only one band should be present. This is fixed in both the download pipeline and dataloader by selecting only the first band.

## Model Architecture

The model uses a branched architecture:
1. Clay branch: Processes Sentinel-2 imagery
2. Static features branch: Processes DEM, DSM, and derived indices
3. Weather branch: Processes time-series weather data 