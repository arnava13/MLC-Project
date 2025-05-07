import torch
import torch.nn.functional as F

def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error computed only over valid grid cells.

    This variant avoids NaNs that arise when unobserved cells in *target* contain
    NaNs by selecting the valid locations *before* performing the arithmetic
    operations rather than multiplying by the mask.
    """
    # Ensure boolean mask
    mask_bool = mask.bool()

    # Select valid elements
    pred_valid = pred[mask_bool]
    target_valid = target[mask_bool]

    num_valid = pred_valid.numel()
    if num_valid == 0:
        # Return a differentiable zero loss if no valid pixels in batch
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    loss = torch.abs(pred_valid - target_valid).mean()
    return loss

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error computed only over valid grid cells, avoiding NaNs."""
    mask_bool = mask.bool()

    pred_valid = pred[mask_bool]
    target_valid = target[mask_bool]

    num_valid = pred_valid.numel()
    if num_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    loss = torch.pow(pred_valid - target_valid, 2).mean()
    return loss 