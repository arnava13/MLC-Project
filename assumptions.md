# Assumptions Made During Development (2025-04-30)

This document lists key assumptions made while integrating the Clay foundation model and debugging the UHI prediction pipeline.

1.  **Clay `patch_size`:**
    *   **Assumption:** The effective patch size used by the pre-trained `clay-v1.5.ckpt` model is **16x16**, despite the checkpoint's `hyper_parameters` dictionary listing `patch_size: 8`.
    *   **Reasoning:** A `ValueError` during the forward pass indicated the model produced 196 patches from a 224x224 input (`196 = (224/16) * (224/16)`), contradicting the expected 784 patches implied by an 8x8 size (`784 = (224/8) * (224/8)`).
    *   **Implementation:** `ClayFeatureExtractor.__init__` in `src/model.py` was explicitly modified to set `self.patch_size = 16`, overriding the value from `hparams`.

2.  **Clay `embed_dim` (Embedding Dimension):**
    *   **Assumption:** The embedding dimension for the `model_size: 'large'` specified in the checkpoint `hyper_parameters` is **1024**.
    *   **Reasoning:** The `'dim'` key was not present in the checkpoint's `hyper_parameters`. 1024 is the standard embedding dimension for ViT-Large models.
    *   **Implementation:** `ClayFeatureExtractor.__init__` in `src/model.py` was modified to infer `embed_dim=1024` when `model_size` is `'large'`.

3.  **Clay Temporal Input (`norm_time`):**
    *   **Assumption:** The Clay model expects temporal metadata as a `(B, 4)` tensor representing sinusoidal encodings: `[sin(week_norm), cos(week_norm), sin(hour_norm), cos(hour_norm)]`.
    *   **Reasoning:** This aligns with standard practices for Transformers and interpretations of the Clay source code (`datamodule.py` referencing `week_norm`, `hour_norm`).
    *   **Implementation:** The `_normalize_clay_timestamp` helper was added to `CityDataSet` in `src/ingest/dataloader.py` to compute this vector, which is returned with the key `'norm_time'`.

4.  **Clay Spatial Input (`norm_latlon`):**
    *   **Assumption:** The Clay model expects spatial location metadata as a `(B, 4)` tensor representing sinusoidal encodings of the *center* coordinates: `[sin(center_lat), cos(center_lat), sin(center_lon), cos(center_lon)]`.
    *   **Reasoning:** A `ValueError` raised by `ClayFeatureExtractor.forward` explicitly stated it expected a `(B, 4)` tensor for the `norm_latlon_tensor` argument, not the `(B, 2, H, W)` grid initially provided. Interpreting the Clay source (`datamodule.py` using `lat_norm`, `lon_norm`) suggests this encoding.
    *   **Implementation:** The `_normalize_clay_latlon` helper in `CityDataSet` (`src/ingest/dataloader.py`) was modified to compute this `(4,)` vector based on the dataset bounds' center, returned with the key `'norm_latlon'`.

5.  **Spectral Band Normalization:**
    *   **Assumption:** The per-band mean/std normalization required by the Clay model is correctly handled within the `ClayFeatureExtractor` class (`src/model.py`).
    *   **Reasoning:** `ClayFeatureExtractor` is designed to load normalization statistics from a `metadata.yaml` file based on the specified `platform` and `bands`.
    *   **Implementation:** Relies on `ClayFeatureExtractor` correctly loading and applying stats from `src/Clay/configs/metadata.yaml` based on the `platform='sentinel-2-l2a'` and `bands=['blue', 'green', 'red', 'nir']` arguments passed during `UHINetCNN` initialization. The `CityDataSet` provides raw band data.

6.  **`target_h_w` for `UHINetCNN.forward`:**
    *   **Assumption:** The `target_h_w` tuple required by `UHINetCNN.forward` should match the height and width of the target UHI grid (`target` tensor) provided by the dataloader.
    *   **Reasoning:** The model needs to know the desired output spatial dimensions to resize its internal prediction before returning.
    *   **Implementation:** The `train_epoch` and `validate_epoch` functions in the notebook now extract `target_unnorm.shape[1:]` and pass it as the `target_h_w` argument to the model. 