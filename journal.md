#Project Journal
## February 25 2025
Searched some related datasets.
### Datasets to Use
#### Corporate Climate Reporting Data
CDP Climate Change Data: Data submitted by companies regarding greenhouse gas (GHG) emissions, energy use, and risk assessments. https://www.cdp.net/en (request access)
#### Corporate Sustainability Reports: Specific GHG emissions and renewable energy usage ratios from companies like Unilever, Coca-Cola, and Microsoft.
Unilever: https://www.unilever.com/sustainability/responsible-business/sustainability-performance-data/　<br>
Coca-cola: https://www.coca-colacompany.com/content/dam/company/us/en/reports/2023-environmental-update/2023-environmental-update.pdf <br>
Microsoft: https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf
#### Remote Sensing Data
OCO-2: CO₂ concentration data taken from satellite images.
https://ocov2.jpl.nasa.gov/science/oco-2-data-center/ <br>
GHGSat: Satellite data monitoring greenhouse gas emissions (methane and CO₂) from specific locations.
https://earth.esa.int/eogateway/missions/ghgsat#data-section (required approval) 

# February 26 2025
### Major difficulty
Some useful datsets require approval from the data provider. <br>
How to adjust data from reports with remote sensing data? <br>
It is difficult to definitively determine whether it is due to the company's location or the nature of its business.<br>
It seems very hard to distinguish the impact of a company's business on the environment due to the geoggraphic scales and other environmental factors.

### Possible directions
Anomaly detection of CO₂ concentration data from reports by using remote sensing data. <br>
Time series forecasting of CO₂ concentration<br>
Evaluate ESC score of companies by using reports and remote sensing data.

### first meeting
Discuss the direction of the project. <br>
Change our plan to follow some public projects. <br>
EY - The 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands: https://challenge.ey.com/challenges/the-2025-ey-open-science-ai-and-data-challenge-cooling-urban-heat-islands-external-participants/data-description <br>

#  February 28 2025
### Datasets
European Sentinel-2 optical satellite data <br>
NASA Landsat optical satellite data <br>

### Additional Datasets
Building footprints of the Bronx and Manhattan regions <br>
Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021

# March 8 2025
## Week 1: Planning

Output Format:

1123 Grid Cells

60 time stamps

Time series model - 1123x1 

Over 60 timestamps

## Input Data:

**Satellite Data**

Satellite images of the area in question, used to derive NDVI, NDWI, NDBI and LST.

Generate median satellite images for our 60 timestamps, ensuring a imaging region that exactly matches the coordinates of our output dataset.

Use NDVI, NDWI, NDBI, LST as our 4 channels and run a convolution over the $W\times H \times4$ image.  (start with this)

“Participants might explore other combinations of bands from the Sentinel-2 and from other satellite datasets as well. For example, you can use mathematical combinations of bands to generate various indices </a> which can then be used as features in your model. These bands or indices may provide insights into surface characteristics, vegetation, or built-up areas that could influence UHI patterns.” 

- Perhaps we use the spectral bands directly instead of NDVI, NDWI, NDBI, LST and these can be our channels?? There are a lot so we would run ablation, starting with those used for NDVI, NDWI, NDBI and LST and then expanding outwards

“Instead of a single point data extraction, participants might explore the approach of creating a focal buffer around the locations (e.g., 50 m, 100 m, 150 m etc). For example, if the specified distance was 50 m and the specified band was “Band 2”, then the value of the output pixels from this analysis would reflect the average values in band 2 within 50 meters of the specific location. This approach might help reduction in error associated with spatial autocorrelation. In this demonstration notebook, we are extracting the band data for each of the locations without creating a buffer zone.” – We should use the resolution of the output dataset i.e **gridcell size** as the area of averaging

**Weather** 

2x locations Bronx & Manhattan

- When predicting for a grid cell, use weather from the closest station

5x columns, merge to 60 timestamps if there are more

—> 5D tensor for every timestamp

**Building Footprint**

- Polygons for buildings
- Perhaps add an embedding of these to the embedding of the final grid cells, so that we can match them exactly on gridcells

**Traffic**

- Research shows car traffic results in UHI, so we are going to use https://developers.google.com/maps/documentation/javascript/examples/layer-traffic to get a scalar traffic density value for each grid cell. Add/concat at near the end of the model when we have grid cells.

# March 15 2025
## Week 2: Planning

Not much progress in this week since we both were very busy for midterms.
try to finish preprocessing for data input and build a simple model (small number of features).

Importamt papers
- FoundTS: Comprehensive and Unified Benchmarking of Foundation Models for Time Series Forecasting <br>
- Deep Time Series Models: A Comprehensive Survey and Benchmark<br>
- N-BEATS: The Unique Interpretable Deep Learning Model for Time Series Forecasting<br>
- TIME-MOE: Billion-Scale Time Series Foundation Models with Mixture of Experts<br>

# March 22 2025:
Computation plans
- Use latest Feature Pyramid Network for image feature <br>
https://unit8co.github.io/darts/index.html#forecasting-models <br>
https://unit8co.github.io/darts/userguide/torch_forecasting_models.html <br>

- Use latest detection/segmentation model for time series (YOLO) <br>

# March 31 2025
Finalised our computational plan
- Extract the tensors specified above (nearly done)
- Use pretrained encoder trained on satellite imagery, will try them all in this order:

1. Clay
2. U-BARN
3. Prithvi

- Use simple LSTM for getting the regression output
- Then try transformers, N-BEATS, and DeepAR etc.

# April 1 2025
Created load_sentinel_LST_tensor.py 
- Loaded Sentinel-2 imagery as a tensor with selected bands using Planetary Computer STAC API.
- Applied cloud cover filtering and bounding box over NYC.
- Loaded Landsat 8 LWIR11 band.
- Resized LST tensor to match Sentinel-2 resolution using bilinear interpolation.
- Combined Sentinel-2 and LST tensors into a final shape of (5, 1448, 1671).
- Confirmed functionality with shape, value range checks, and LST heatmap visualization.

# April 11 2025: Began to create dataloaders 
- We want a dataloader for each City (DONE)
- Each dataloader consolidates the UHI data to a grid based on an input resolution parameters (DONE)
    TODO:
- Create a function to get weather data nearest to each latlong position of the UHI data (DONE).
- loads the satellite data for the bounding box of the city and the time window specified by the UHI
  data, and then we get the median mosaic (cloudless) of the satellite data in some averaging window
  before the UHI observation day, will start with 10 days. (nearly done)
- load the weather data nearest every UHI observation point using API (nearly done)
- create the lst data as a tensor with the same grid as the UHI data.

Goal for Tuesday the 15th:
- Get a simple model working, we will use the Clay forward pass just once as a feature extractor and
  not backpropagate through it. Then we will combine all features in an LSTM and train that.
- Trying to build in such a way that we don't load all of the data into memory at once, but rather
  query APIs when we do __getitem__() in the dataset class, and then pass this to a standard pytorch
  dataloader.

# April 15 2025: Completed the dataloader

## `ingest/get_median.py`: Data extraction functions
- Implemented `load_sentinel_tensor_from_bbox_median()`: 
  - Loads and computes the median composite of Sentinel-2 imagery over a given bounding box and time window.
- Implemented `load_lst_tensor_from_bbox_median()`:
  - Loads and computes the median of LST over the same region and time.
- Resized LST tensors to match Sentinel resolution.

## Modified `CityDataSet` class 
- Built a PyTorch `Dataset` class that:
  - Loads Sentinel-2 tensors median over a time window per UHI.
  - Associates each UHI timestamp with its corresponding satellite data window.

## Integrated weather data from Open-Meteo
- Used pre-downloaded `nyc_weather_grid.csv` from Open-Meteo API (`weather_data.py`).
- Matched weather data (max/min temperature, precipitation) based on:
  - Rounded lat/lon or tolerance-based proximity
  - Normalized date (from timestamp)

## Integrated LST as optional input (`include_lst=True`)
- Added a boolean flag `include_lst` in `CityDataSet`.
- If True:
  - Loads LST median tensor for each UHI observation.
  - Resizes LST tensor to match Sentinel resolution.
  - Concatenates LST as an additional band → Final shape: `(5, H, W)`  
- If False:
  - Sentinel tensor shape remains `(4, H, W)`


## April 26 2025:

- Modfiied the dataloder to load files from disk, using new download_data.ipynb
- Added a new function to load the UHI data from disk.
- Added a new function to load the weather data from disk.
- Created model.py to load the pretained clay model as the feature map extractor and added a
  temporal conv net to extract features from the time series data.
- Next steps:
  - Verify data downloads and download satellite images for NYC successfully on GCP
  runtime.
  - Train the model:
    - For the losses we want to mask away losses for missing values and for gridcells that were not
      measured. This allows us to mantain a consistent output shape. (Needs modification to the model.)
    - Try to use the model to predict the UHI data for NYC.

- Create a presentation video and slides for the project.

## April 29 2025:
- New deep learning based model acheives positive R2 scores on the UHI data

## April 30 2025:
src/uhi-pipeline
- Collected and prepared UHI dataset (`uhi.csv`) with latitude, longitude, datetime, and UHI Index

- Extracted environmental features:
  - NDVI, NDWI, NDBI from Sentinel-2 satellite imagery
  - LST (Land Surface Temperature) from Landsat-8 imagery

- Integrated weather data (`manhattan_weather.csv`, `bronx_weather.csv`):
  - Merged Bronx and Manhattan Mesonet station data using inverse distance weighting (IDW)
  - Included air temperature, relative humidity, wind speed, wind direction (as sin/cos), and solar flux

- Incorporated urban form features (`Building_Footprint.kml`):
  - Computed building footprint metrics (building count, total area) within 100m buffer around each UHI point using KML data

- Built predictive models:
  - Random Forest
  - XGBoost
  - MLP (Multi-Layer Perceptron) with increasing complexity

- Evaluated model performance:
  - Random Forest achieved R² ≈ 0.95
  - XGBoost achieved R² ≈ 0.92
  - MLP improved from R² ≈ -0.67 to ≈ 0.62 with deeper networks

- Performed feature importance analysis:
  - Random Forest and XGBoost feature importance
  - SHAP analysis for both Random Forest and XGBoost models

- Next steps:
  - add band features to the model (not the ND indices)


## May 5 2025:

- Informed by other people's comments on the EY forum and by Shunsuke's implementation of the
  RF/XGBoost/MLP model, we have worked on a new branched UHI model architecure
  (dataloader_branched.py , branched_uhi_model.py).

- Processes temporal weather features with a ConvLSTM before concatenating (flag-configurable)
  static features to channel dimmension of final hidden state from the ConvLSTM. 

- Passes feature cube to a UNet-like architecture to make UHI predictions at each timestamp.