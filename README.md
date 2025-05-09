# Urban Heat Island (UHI) Prediction Project

This project aims to predict Urban Heat Island (UHI) intensity using machine learning techniques, leveraging a diverse set of geospatial and temporal data sources. The primary goal is to develop robust models that can accurately capture the complex interplay of factors contributing to UHI effects in urban environments.

## Project Structure

-   **`data/`**: (Not versioned) Stores raw and processed data, including UHI measurements, weather station records, satellite imagery mosaics (Sentinel-2), elevation models (DEM/DSM), LST, and derived spectral indices.
-   **`notebooks/`**: Contains Jupyter notebooks for various tasks:
    -   `download_data.ipynb`: Initial data acquisition and preprocessing scripts.
    -   `RF & XGBoost.ipynb`: Experiments with Random Forest, XGBoost, and MLP baselines.
    -   `train_uhi_cnn.ipynb`: Training pipeline for the CNN-based UHI model (`UHINetCNN`).
    -   `train_uhi_branched_model.ipynb`: Training pipeline for the branched recurrent UHI model (`BranchedUHIModel`).
-   **`src/`**: Houses the core Python source code for the project.
    -   **`Clay/`**: Contains the Clay foundation model (geospatial foundation model) for extracting features from Sentinel-2 imagery.
    -   **`ingest/`**: Modules for data loading and preprocessing.
        -   `data_utils.py`: Common utilities for data transformation, resampling, normalization, and grid calculations.
        -   `dataloader_cnn.py`: PyTorch `Dataset` and `DataLoader` for the `UHINetCNN` model.
        -   `dataloader_branched.py`: PyTorch `Dataset` and `DataLoader` for the `BranchedUHIModel`.
    -   **`model.py`**: Defines the `UHINetCNN` architecture and supporting components like U-Net blocks and the Clay feature extractor wrapper.
    -   **`branched_uhi_model.py`**: Defines the `BranchedUHIModel` architecture, including the ConvLSTM cell and network, and different head options.
    -   **`train/`**: Utilities for the training process.
        -   `loss.py`: Custom loss functions (e.g., `masked_mse_loss`, `masked_mae_loss`).
        -   `train_utils.py`: Generic training and validation epoch functions, metric calculations, UHI statistics, and other training helpers.
-   **`models_to_train.md`**: A checklist outlining the planned sequence of model training experiments.
-   **`README.md`**: This file.
-   **`requirements.txt`**: Python package dependencies.

## Data Overview and Preprocessing

The project utilizes several data sources for New York City (NYC), primarily focusing on the summer period of 2021 (June 1st to September 1st):

1.  **UHI Target Data**: Point-based UHI index measurements from `uhi.csv`.
2.  **Weather Data**: Hourly weather station data (air temperature, relative humidity, wind speed/direction, solar flux) from two stations (Bronx and Manhattan). These are interpolated to UHI point locations using Inverse Distance Weighting (IDW) for baseline models, and gridded for deep learning models. Wind direction is decomposed into sine and cosine components.
3.  **Satellite Imagery & Derived Products**:
    *   **Sentinel-2 L2A**: Used to create a cloudless median mosaic for the specified time window. Bands include Blue, Green, Red, NIR, SWIR16, SWIR22.
        *   **Spectral Indices**: NDVI, NDBI, NDWI are calculated from the Sentinel-2 mosaic.
        *   **Clay Features**: The pre-trained Clay foundation model (`clay-v1.5.ckpt`) is used to extract high-level features from Sentinel-2 imagery (Blue, Green, Red, NIR bands).
    *   **Landsat 8**: Used to derive Land Surface Temperature (LST) data. A median LST product is generated.
4.  **Elevation Data (USGS 3DEP ~10m)**:
    *   **Digital Elevation Model (DEM)**: Represents bare earth elevation.
    *   **Digital Surface Model (DSM)**: Represents surface elevation including buildings and vegetation.
    *   Both are resampled and normalized using robust percentile-based clipping.
5.  **Building Footprints**: KML data used in baseline models to calculate building count and area within a 100m radius of UHI points.

**Key Preprocessing Steps for Deep Learning Models**:
*   **Common Feature Resolution**: All spatial input features (weather grids, DEM, DSM, LST, Clay features, indices, direct Sentinel bands) are resampled to a common configurable spatial resolution (e.g., 30m) before being fed into the models. The UHI target grid is maintained at its own resolution (e.g., 10m).
*   **Normalization**:
    *   DEM/DSM: Robust percentile-based normalization (2nd-98th percentile) to scale values between 0 and 1.
    *   LST: Normalized based on predefined min/max or statistics.
    *   Weather: Standardized using pre-calculated means and standard deviations.
    *   Target UHI: For training deep learning models, the target UHI values are Z-score normalized using statistics calculated from the training set. Predictions are then un-normalized for metric calculation.
*   **Data Loaders**: `CityDataSetCNN` and `CityDataSetBranched` handle on-the-fly resampling, feature selection based on flags, and batching for PyTorch models.

## Models and Experiments

### 1. Baseline Models (`notebooks/RF_XGBoost.ipynb`)

Traditional machine learning models were implemented as baselines using point-based features:
*   **Features Used**: NDVI, NDWI, NDBI, LST, interpolated weather variables (air\_temp, rel\_humidity, avg\_windspeed, wind\_dir\_sin, wind\_dir\_cos, solar\_flux), building\_count\_100m, building\_area\_100m.
*   **Random Forest Regressor**: Achieved **R² = 0.9486**.
*   **XGBoost Regressor**: Achieved **R² = 0.9260**.
*   **MLP Regressor**: Best R² = **0.6169** (with layers 512, 256, 128, 64, 32 and scaled features).
*   **Ensemble (Weighted Average)**: RF (0.5) + XGBoost (0.4) + MLP (0.1) achieved **R² = 0.9390**.

### 2. Deep Learning Models

#### a. CNN Model (`UHINetCNN` in `src/model.py`)

*   **Architecture**: A U-Net style Convolutional Neural Network.
*   **Inputs**:
    *   Single timestamp weather grid.
    *   Static features: DEM, DSM, LST, spectral indices (NDVI, NDBI, NDWI), Clay features, and optionally, specified raw Sentinel-2 bands.
*   **Key Features**:
    *   Integrates the pre-trained Clay model as a feature extractor.
    *   All inputs are resampled to a common feature resolution.
    *   U-Net decoder directly predicts the UHI grid at the target UHI resolution.
*   **Training**: `notebooks/train_uhi_cnn.ipynb`.

#### b. Branched Recurrent Model (`BranchedUHIModel` in `src/branched_uhi_model.py`)

*   **Architecture**: A two-branch model:
    1.  **Temporal Branch**: Uses a ConvLSTM network to process sequences of weather grids.
        *   ConvLSTM cells include orthogonal weight initialization and forget gate bias set to 1.0 for improved stability.
        *   Temporal Pooling: Currently uses "Global Timestep Weights" where learnable weights are applied to the ConvLSTM outputs at each timestep, followed by a weighted sum. Previous explorations included using only the last hidden state and a learnable exponential decay.
    2.  **Static Branch**: Processes static features.
        *   Clay features (from Sentinel-2 mosaic).
        *   Other static features: DEM, DSM, LST, spectral indices (NDVI, NDBI, NDWI), and optionally, specified raw Sentinel-2 bands.
*   **Fusion**: Features from both branches are projected to a common channel dimension and then concatenated.
*   **Head**: The fused features are passed to a prediction head.
    *   **U-Net Head**: A U-Net style decoder (`UNetDecoderWithTargetResize`).
    *   **Simple CNN Head**: A simpler CNN with a few convolutional blocks (`SimpleCNNHead`).
*   **Key Features**:
    *   Explicitly models temporal dependencies in weather data.
    *   Flexible head architecture.
    *   All inputs (per timestep for weather, once for static) are at a common feature resolution.
*   **Training**: `notebooks/train_uhi_branched_model.ipynb`.

## Key Architectural Decisions and Evolution

*   **Common Feature Resolution**: Adopted to simplify model architecture and ensure consistent spatial alignment of diverse input features before they enter the neural network.
*   **Target Normalization**: Shifted to normalizing the target UHI variable (Z-scoring) directly before loss calculation in the training loop, and un-normalizing predictions for metrics. This helps stabilize training and makes loss values more interpretable relative to normalized targets.
*   **DEM/DSM Handling**: Implemented robust percentile-based normalization after encountering issues with raw value ranges and potential outliers. Addressed multi-band issues from data sources by selecting only the first band.
*   **Sentinel-2 Bands**: Raw Sentinel-2 bands (distinct from Clay input) can now be optionally included as separate features, controlled by the `use_sentinel_composite` flag and `sentinel_bands_to_load` list in the dataloader configuration. If `use_sentinel_composite` is true and `sentinel_bands_to_load` is non-empty, specified bands are added.
*   **ConvLSTM Enhancements**:
    *   Implemented orthogonal initialization for ConvLSTM weights and set forget gate bias to 1.0 to aid learning long-term dependencies.
    *   Explored several temporal pooling mechanisms for the ConvLSTM output sequence, settling on learnable global timestep weights for current experiments.
*   **Modular Training Utilities**: Refactored training and validation loops into generic functions (`train_epoch_generic`, `validate_epoch_generic` in `src/train/train_utils.py`) to support different model architectures and batch structures more robustly.
*   **Hyperparameter Adjustments**: Iteratively tuned learning rates, weight decay, and ConvLSTM hidden dimensions based on experimental results.

## Current Training Plan

The immediate training plan is detailed in `models_to_train.md`. It focuses on systematically evaluating the impact of DEM/DSM, the ConvLSTM recurrence, and different head architectures, before exploring ablations like removing Clay features in favor of spectral indices.

## Known Issues / Areas for Improvement

*   **Multi-band Elevation Data**: While handled in the download and dataloading pipeline by selecting the first band, the root cause in the Planetary Computer 3DEP collection (sometimes returning 5 bands) is external.
*   **Computational Resources**: Training deep learning models, especially with ConvLSTM and high-resolution inputs, can be computationally intensive.
*   **Hyperparameter Optimization**: Further systematic hyperparameter tuning could yield performance improvements.
