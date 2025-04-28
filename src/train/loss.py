import torch
import torch.nn.functional as F

def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculates the Mean Absolute Error (MAE) only on valid (masked) grid cells.

    Args:
        pred (torch.Tensor): Predicted UHI grid (B, H, W).
        target (torch.Tensor): Ground truth UHI grid (B, H, W).
        mask (torch.Tensor): Boolean or float mask (B, H, W), where 1 indicates valid cells.

    Returns:
        torch.Tensor: Scalar MAE loss computed over valid cells.
    """
    # Ensure mask is float for multiplication
    mask = mask.float()
    # Calculate absolute difference
    diff = torch.abs(pred - target)
    # Apply mask
    masked_diff = diff * mask
    # Calculate sum over masked elements
    total_mae = masked_diff.sum()
    # Count number of valid elements
    num_valid = mask.sum()
    # Avoid division by zero if mask is all zeros
    if num_valid == 0:
        # Return 0 loss or handle as appropriate (e.g., return NaN, raise error)
        # Returning 0 is common if it's possible to have batches with no valid data
        return torch.tensor(0.0, device=pred.device, requires_grad=True) # Ensure it's on the right device and differentiable

    # Return mean absolute error
    return total_mae 

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculates the Mean Squared Error (MSE) only on valid (masked) grid cells.

    Args:
        pred (torch.Tensor): Predicted UHI grid (B, H, W).
        target (torch.Tensor): Ground truth UHI grid (B, H, W).
        mask (torch.Tensor): Boolean or float mask (B, H, W), where 1 indicates valid cells.

    Returns:
        torch.Tensor: Scalar MSE loss computed over valid cells.
    """
    mask = mask.float()
    diff_sq = (pred - target) ** 2
    masked_diff_sq = diff_sq * mask
    total_mse = masked_diff_sq.sum()
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return total_mse 