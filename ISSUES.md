# Project Issues

## Issue 1: GPU Out Of Memory (OOM) during Training

**Reported:** 2024-07-29
**Status:** Investigating

**Description:**
User reported encountering a GPU Out Of Memory error after modifications to incorporate higher-resolution DEM/DSM data and potentially new model branches. The exact cause (batch size, feature resolution, model complexity, lingering high-res data loading) is under investigation.

**Affected Files/Modules:**
- `notebooks/train_uhi_cnn.ipynb` OR `notebooks/train_uhi_branched_model.ipynb`
- `src/ingest/dataloader_cnn.py` / `src/ingest/dataloader_branched.py`
- `src/model.py` / `src/branched_uhi_model.py`

**Attempted Fixes:**
- (2024-07-29) Suggested halving the `batch_size` in the relevant training notebook configuration. (Pending user confirmation)
- (2024-07-29) Suggested increasing `feature_resolution_m` (coarser resolution) if batch size reduction is insufficient. (Pending user confirmation) 