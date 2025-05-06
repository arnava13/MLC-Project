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

### NaN Loss Values during Training (2025-05-06)

**Reported:** 2025-05-06
**Status:** Resolved

**Description:**
Training runs were resulting in NaN loss values, causing the training to stop after a few iterations. This indicated numerical instability in the model training.

**Root Cause:**
Several factors contributed to this issue:
1. The loss function was calculating mean by dividing a potentially large sum by a small number of valid pixels
2. The ConvLSTM weights were not properly initialized, potentially leading to exploding gradients
3. Lack of gradient clipping allowed gradients to grow too large during backpropagation

**Affected Files/Modules:**
- `src/train/loss.py`
- `src/train/train_utils.py`  
- `src/branched_uhi_model.py`

**Solution:**
1. Fixed the loss functions in `src/train/loss.py` to properly calculate mean values
2. Added gradient clipping in `train_utils.py` with a max norm of 1.0
3. Added orthogonal initialization to ConvLSTM cell weights in `branched_uhi_model.py`
4. Set forget gate bias to 1.0 to improve learning stability

### Channel mismatch in model input for DEM/DSM data (2023-05-06)

**Issue Description:**
The model expected 1026 input channels (1024 from Clay features + 2 from DEM/DSM), but was receiving 1030 channels (1024 from Clay + 6 from other features). This caused a shape mismatch error during model forward pass.

**Root Cause:**
DEM and DSM files were being loaded with 5 bands each instead of 1, resulting in 10 channels where only 2 were expected. This was confirmed by examining the model logs: "DSM has 5 bands, using only the first band" warning.

**Investigation:**
1. From the source website, both DEM and DSM data should only have a single band.
2. Examining logs showed: "DSM has 5 bands, using only the first band" warning.
3. In the model, we confirmed a channel mismatch error where static_proj expected 1026 channels but got 1030.

**Solution:**
1. Modified `dataloader_branched.py` to explicitly select only the first band from each elevation file
2. Added debug logs to help identify the exact shapes at each processing step
3. Suppressed frequent warning messages during training to reduce noise

**Verification:**
After the changes, the dataloader now correctly provides 2 channels (1 for DEM, 1 for DSM) instead of 10. This resolved the channel mismatch error in the model forward pass.

**Status:** Resolved 