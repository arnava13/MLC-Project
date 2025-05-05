# Project Issues

## Issue 1: GPU Out Of Memory (OOM) during Training

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