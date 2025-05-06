# Models to Train: UHI Prediction

This document outlines the planned sequence of model training experiments.

## Phase 1: Validate DEM/DSM Impact with CNN

-   [ ] **Model 1: CNN + DEM/DSM + Clay (U-Net Head)**
    -   **Objective:** Establish if DEM/DSM features, when processed correctly with Clay and a U-Net head, yield improved R² compared to previous simpler CNNs.
    -   **Key Configuration:**
        -   Model: `UHINetCNN`
        -   Features: Weather, DEM, DSM, Clay
        -   Head: U-Net
        -   Dataloader: `src/ingest/dataloader_cnn.py`
        -   Notebook: `notebooks/train_uhi_cnn.ipynb`
    -   **Expected Outcome:** Positive R² and smoother validation metric progression.

## Phase 2: Introduce Recurrence (ConvLSTM)

*Condition: Proceed if Model 1 shows promising R² (e.g., significantly above 0).*

-   [ ] **Model 2: Branched ConvLSTM + DEM/DSM + Clay (Simple CNN Head)**
    -   **Objective:** Test the ConvLSTM branch for temporal weather processing alongside static features, starting with a simpler head to isolate ConvLSTM performance.
    -   **Key Configuration:**
        -   Model: `BranchedUHIModel`
        -   Features: Weather (sequence), DEM, DSM, Clay
        -   Temporal Pooling: Global Timestep Weights
        -   ConvLSTM Hidden Dims: `[32, 16]`
        -   Head: Simple CNN
        -   Dataloader: `src/ingest/dataloader_branched.py`
        -   Notebook: `notebooks/train_uhi_branched_model.ipynb`
    -   **Expected Outcome:** Stable training, further R² improvement over Model 1.

-   [ ] **Model 3: Branched ConvLSTM + DEM/DSM + Clay (U-Net Head)**
    -   **Objective:** If Model 2 is stable and shows R² improvement, switch to the more complex U-Net head to potentially capture finer spatial details.
    -   **Key Configuration:**
        -   Model: `BranchedUHIModel`
        -   Features: Weather (sequence), DEM, DSM, Clay
        -   Temporal Pooling: Global Timestep Weights
        -   ConvLSTM Hidden Dims: `[32, 16]`
        -   Head: U-Net
        -   Dataloader: `src/ingest/dataloader_branched.py`
        -   Notebook: `notebooks/train_uhi_branched_model.ipynb`
    -   **Expected Outcome:** Best R² performance so far.

## Phase 3: Explore Feature Ablation (No Clay)

*Condition: Proceed if Model 3 trains successfully to convergence with good R².*

-   [ ] **Model 4: Branched ConvLSTM + DEM/DSM + Indices (No Clay, Simple CNN Head)**
    -   **Objective:** Evaluate if spectral indices (NDVI, NDBI, NDWI) can serve as a lighter-weight alternative to the Clay backbone, using the ConvLSTM architecture. Start with a simple CNN head.
    -   **Key Configuration:**
        -   Model: `BranchedUHIModel`
        -   Features: Weather (sequence), DEM, DSM, NDVI, NDBI, NDWI (enable these flags, disable `use_clay`)
        -   Temporal Pooling: Global Timestep Weights
        -   ConvLSTM Hidden Dims: `[32, 16]`
        -   Head: Simple CNN
        -   Dataloader: `src/ingest/dataloader_branched.py`
        -   Notebook: `notebooks/train_uhi_branched_model.ipynb` (adjust feature flags in config)
    -   **Expected Outcome:** Determine performance trade-off without Clay.

-   [ ] **Model 5: Branched ConvLSTM + DEM/DSM + Indices (No Clay, U-Net Head)**
    -   **Objective:** If Model 4 shows promise, test with the U-Net head.
    -   **Key Configuration:**
        -   Model: `BranchedUHIModel`
        -   Features: Weather (sequence), DEM, DSM, NDVI, NDBI, NDWI
        -   Temporal Pooling: Global Timestep Weights
        -   ConvLSTM Hidden Dims: `[32, 16]`
        -   Head: U-Net
        -   Dataloader: `src/ingest/dataloader_branched.py`
        -   Notebook: `notebooks/train_uhi_branched_model.ipynb` (adjust feature flags in config)
    -   **Expected Outcome:** Assess best performance achievable with indices instead of Clay.

## General Notes:

*   Monitor `val_r2`, `val_loss`, and `val_rmse` closely.
*   Ensure `wandb` logging is active for all runs.
*   Adjust hyperparameters (learning rate, weight decay, patience) as needed based on observations from initial epochs of each model.
*   The primary goal is to achieve stable training with R² > 0.5, ideally approaching or exceeding 0.7. 