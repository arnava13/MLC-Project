# Project Issues

## Active Issues

### GPU Out Of Memory (OOM) during Training

**Reported:** 2024-07-29
**Status:** Investigating

**Description:**
User reported encountering a GPU Out Of Memory error after modifications to incorporate higher-resolution DEM/DSM data and potentially new model branches. The exact cause (batch size, feature resolution, model complexity, lingering high-res data loading) is under investigation. User confirmed `batch_size` was already set to 2 in `notebooks/train_uhi_branched_model.ipynb`.

**Affected Files/Modules:**
- `notebooks/train_uhi_branched_model.ipynb` (primarily)
- `notebooks/train_uhi_cnn.ipynb` OR `notebooks/train_uhi_branched_model.ipynb`
- `src/ingest/dataloader_cnn.py` / `src/ingest/dataloader_branched.py`
- `src/model.py` / `src/branched_uhi_model.py`

**Attempted Fixes:**
- (2024-07-29) User confirmed `batch_size` is already 2 (later logs show effective train batch size is 1).
- (2024-07-29) Suggested increasing `feature_resolution_m` in the training notebook config (e.g., from 10m to 20m or 30m) as the next step. (User tried 30m, still likely OOM).
- (2024-07-29) User further modified config (`train_uhi_branched_model.ipynb`, cell 4): 
    - Increased `feature_resolution_m` to 50m.
    - Reduced `convlstm_hidden_dims` to `[16, 8]`.
    - Reduced `unet_base_channels` to 32.
    - Reduced `unet_depth` to 3.
    - Reduced projection channels (`proj_static_ch`, `proj_temporal_ch`) to 8.
    - Reduced `num_workers` to 1.
    - Set `n_train_batches` to 47 (ensuring train batch size 1).
    - Set `uhi_grid_resolution_m` to 5m.
    (Pending user confirmation of OOM resolution).
- (2024-07-29) Secondary suggestions: Reduce model complexity (ConvLSTM hidden dims, U-Net base channels/depth). (Partially addressed in latest config change). 

## Resolved Issues

### NaN Loss Values During Training (2023-05-06)

**Issue Description:**
During model training, loss values quickly became NaN, causing training to fail. This was observed in the logs: "Training: 2%|▏ | 1/47 [00:09<07:06, 9.28s/it, loss=nan]"

**Root Cause:**
The loss functions `masked_mse_loss` and `masked_mae_loss` in `src/train/loss.py` were returning the sum of errors instead of the mean. Without dividing by the number of valid elements, the loss values could become very large and lead to numerical instability, especially with larger batch sizes or deeper models.

**Fix:**
Modified both loss functions to correctly return the mean error by dividing by the number of valid elements:
```python
# Before:
return total_mse  # Just returning sum of all squared errors

# After:
return total_mse / num_valid  # Properly calculating mean
```

**Status:** Resolved

### Channel mismatch in model input for DEM/DSM data (2023-05-06)

**Issue Description:**
The model expected 1026 input channels (1024 from Clay features + 2 from DEM/DSM), but was receiving 1030 channels (1024 from Clay + 6 from other features). This caused a shape mismatch error during model forward pass.

**Root Cause:**
DEM and DSM files were being loaded with 5 bands each instead of 1, resulting in 10 channels where only 2 were expected. This was confirmed by examining the model logs: "[DEBUG DATALOADER] combined_static_features shape: (6, 373, 323)" and "Included static features: ['dem', 'dsm']".

**Investigation:**
1. From the source website https://planetarycomputer.microsoft.com/dataset/3dep-lidar-dsm#overview, both DSM and DEM data should only have one band.
2. Examining logs showed: "DSM has 5 bands, using only the first band" warning.
3. Files were downloaded using stackstac and saved as Cloud Optimized GeoTIFF (COG) files.

**Fix:**
1. Modified the dataloader code to consistently select only the first band from each elevation file if multiple bands were detected:
```python
if dem_feat_res.shape[0] > 1:
    logging.warning(f"DEM has {dem_feat_res.shape[0]} bands, using only the first band.")
    dem_feat_res = dem_feat_res[0:1]  # Keep as 3D with a single channel
```

2. Updated the download_data.ipynb notebook to ensure only single-band data is saved during the download process:
```python
# Check if data has multiple bands/channels and take only the first one
if len(dem_data.shape) > 2 and dem_data.shape[0] > 1:
    logging.warning(f"DEM data has {dem_data.shape[0]} bands when it should have 1. Keeping only first band.")
    dem_data = dem_data[0:1]
```

**Status:** Resolved 