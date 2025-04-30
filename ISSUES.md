# Project Issues

This document tracks open questions and potential issues during development.

## Open Issues (as of Initial Roadmap)

1.  **Target Grid Resolution Alignment:** Confirm that the target UHI grid resolution (derived from `uhi.csv`'s `x_grid`, `y_grid`) aligns correctly with the potentially resized weather grid after zooming in the dataloader. Mismatches could lead to incorrect loss calculation.
2.  **Clay Feature Fine-tuning:** Decide whether to keep the precomputed Clay features entirely frozen or allow fine-tuning of the final projection layer (`proj` in `UHINetConvGRU`). Freezing is simpler and faster initially, but fine-tuning might improve performance if static features aren't perfectly adapted.
3.  **Memory Footprint of Time Embeddings:** Evaluate the memory usage when broadcasting the 4-channel time embedding map to the full spatial dimensions (H, W), especially for high-resolution grids. If memory becomes an issue, consider alternative approaches like concatenating time features only at the regressor input.

## Resolved Issues

1.  **`TypeError: object() takes no arguments` / State Dictionary Key Mismatch during `UHINetCNN` Clay backbone initialization (Multiple Attempts)**
    *   **Timestamp:** 2025-04-28/29
    *   **File:** `src/model.py` (`ClayFeatureExtractor.__init__`)
    *   **Root Cause:** Initial attempts using `ClayMAEModule.load_from_checkpoint` failed due to internal conflicts with LightningCLI/ArgumentParser triggered by the `_instantiator` hyperparameter. Manual loading attempts then revealed state dictionary key mismatches (missing `model.` prefix) and incorrect target object for `load_state_dict`.
    *   **Resolution:** Implemented manual loading: `torch.load` checkpoint, prepare `hparams` (overriding `metadata_path` with absolute path, removing `_instantiator`), instantiate `ClayMAEModule`, adjust state dict keys (remove `model.` prefix), load adjusted state dict into `self.model.model` (the nested `ClayMAE` instance) using `strict=False`.

## Error Log & Attempted Fixes

### 3. State Dictionary Key Mismatch during Manual Loading

*   **Timestamp:** 2025-04-29
*   **File:** `src/model.py`
*   **Context:** When using manual loading, `load_state_dict(strict=False)` reported missing encoder/decoder keys.
*   **Diagnosis:** Debug prints showed that the instantiated `ClayMAEModule` (`self.model`) expects keys starting with `model.`, while the raw checkpoint state dict also has keys starting with `model.`. The prefix removal logic (`k.replace("model.", ...)`) was creating incorrect keys (`encoder...`) for loading into `self.model`.
*   **Attempted Fix (2025-04-29):** Modified manual loading in `ClayFeatureExtractor.__init__` to remove the `model.` prefix from checkpoint keys but load the resulting `adjusted_state_dict` into the nested `self.model.model` object.
*   **Status:** *Successful (Initialization now works)*

### 4. `NameError: name 'target_timestamp' is not defined` in DataLoader worker
*   **Timestamp:** 2025-04-29
*   **File:** `src/ingest/dataloader.py` (`CityDataSet.__getitem__`)
*   **Context:** During training loop, worker process fails with `NameError` when calling `self._get_time_embedding(target_timestamp)`.
*   **Hypothesis:** Although `target_timestamp` is defined earlier, an unexpected code path or state issue within the worker leads to the variable being undefined or having an incorrect type/value at the point of the call.
*   **Attempted Fix (2025-04-29):** Added debug prints and type checking for `target_timestamp` immediately before the call to `_get_time_embedding`, removed surrounding `try/except`.
*   **Status:** *Pending validation* 