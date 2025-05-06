# Project Issues

## Active Issues

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
