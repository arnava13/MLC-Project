# Project Issues

This document tracks open questions and potential issues during development.

## Open Issues (as of Initial Roadmap)

1.  **Target Grid Resolution Alignment:** Confirm that the target UHI grid resolution (derived from `uhi.csv`'s `x_grid`, `y_grid`) aligns correctly with the potentially resized weather grid after zooming in the dataloader. Mismatches could lead to incorrect loss calculation.
2.  **Clay Feature Fine-tuning:** Decide whether to keep the precomputed Clay features entirely frozen or allow fine-tuning of the final projection layer (`proj` in `UHINetConvGRU`). Freezing is simpler and faster initially, but fine-tuning might improve performance if static features aren't perfectly adapted.
3.  **Memory Footprint of Time Embeddings:** Evaluate the memory usage when broadcasting the 4-channel time embedding map to the full spatial dimensions (H, W), especially for high-resolution grids. If memory becomes an issue, consider alternative approaches like concatenating time features only at the regressor input.

## Resolved Issues

*(None yet)* 