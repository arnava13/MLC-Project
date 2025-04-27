import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import logging # Added logging

# Need to import the ClayFeatureExtractor class itself
from src.model import ClayFeatureExtractor

# Helper ConvGRU Cell (from src.model)
class ConvGRUCell(nn.Module):
    """Single-layer ConvGRU with 3x3 kernels."""
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # Convolution for Z (update) and R (reset) gates
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size, padding=padding)
        # Convolution for the candidate hidden state h~
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: input tensor (B, C_in, H, W)
        # h: hidden state from previous time step (B, C_hid, H, W)
        combined = torch.cat([x, h], dim=1) # Concatenate along channel dimension

        # Calculate Z and R gates
        zr = torch.sigmoid(self.conv_zr(combined))
        z, r = torch.chunk(zr, 2, dim=1) # Split into two tensors

        # Calculate candidate hidden state h~
        combined_r = torch.cat([x, r * h], dim=1) # Element-wise multiplication for reset gate
        h_tilde = torch.tanh(self.conv_h(combined_r))

        # Calculate next hidden state
        h_next = (1 - z) * h + z * h_tilde # Update gate combines old state and candidate state
        return h_next

class UHINetConvGRU(nn.Module):
    """
    Spatio-temporal model using a cloudless mosaic processed dynamically by Clay,
    combined with weather grids, time embeddings, and optional dynamic LST.

    Predicts a full UHI grid.

    Args:
        mosaic_channels (int): Number of channels in the input cloudless mosaic (e.g., 4 for RGB+NIR).
        weather_channels (int): Number of channels in the weather grid (e.g., 3: max/min/precip).
        time_emb_channels (int): Number of channels in the time embedding map (e.g., 4: sin/cos day/minute).
        lst_channels (int): Number of channels for dynamic LST (1 if included, 0 otherwise).
        clay_embed_dim (int): Output dimension of the Clay encoder (typically 768).
        proj_ch (int): Number of channels to project Clay features down to.
        hid_ch (int): Number of hidden channels in the ConvGRU.
        freeze_clay (bool): Whether to freeze the weights of the Clay encoder.
    """
    def __init__(self,
                 mosaic_channels: int,
                 weather_channels: int = 3,
                 time_emb_channels: int = 4,
                 lst_channels: int = 1,
                 clay_embed_dim: int = 768, # Added Clay output dim
                 proj_ch: int = 64,
                 hid_ch: int = 64,
                 freeze_clay: bool = True):
        super().__init__()
        self.mosaic_channels = mosaic_channels
        self.weather_channels = weather_channels
        self.time_emb_channels = time_emb_channels
        self.lst_channels = lst_channels
        self.clay_embed_dim = clay_embed_dim
        self.proj_ch = proj_ch
        self.hid_ch = hid_ch
        self.freeze_clay = freeze_clay

        # Instantiate Clay Encoder within the model
        self.clay_encoder = ClayFeatureExtractor(in_chans=mosaic_channels, freeze=freeze_clay)

        # Static Feature Path
        # Projects high-dimensional Clay features down to a manageable size
        self.static_proj = nn.Conv2d(clay_embed_dim, proj_ch, kernel_size=1)

        # Initial Hidden State Generator
        # Uses projected static features to initialize the ConvGRU hidden state
        # Can be a simple 1x1 conv or identity if proj_ch == hid_ch
        self.init_map = nn.Conv2d(proj_ch, hid_ch, kernel_size=1) if proj_ch != hid_ch else nn.Identity()

        # Dynamic Input Path
        # ConvGRU processes the time-varying inputs (weather, time embeddings, LST)
        dynamic_input_channels = weather_channels + time_emb_channels + lst_channels
        self.gru = ConvGRUCell(dynamic_input_channels, hid_ch)

        # Output Regressor
        # Predicts the final UHI grid from the ConvGRU's hidden state
        self.regressor = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch // 2, 1, kernel_size=1)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Processes a batch dictionary from the CityDataSet.

        Args:
            batch (dict): A dictionary containing:
                - 'cloudless_mosaic': (B, C_mosaic, H_mosaic, W_mosaic)
                - 'weather_seq': (B, T, C_w, H, W)
                - 'time_emb_seq': (B, T, C_t, H, W)
                - 'lst_seq': (B, T, C_l, H, W)
                - 'target': (B, H, W) - Not used in forward pass
                - 'mask': (B, H, W) - Not used in forward pass

        Returns:
            torch.Tensor: Predicted UHI grid (B, H, W).
        """
        cloudless_mosaic = batch['cloudless_mosaic']
        weather_seq = batch['weather_seq']
        time_emb_seq = batch['time_emb_seq']
        lst_seq = batch['lst_seq']

        # Target grid size from dynamic data
        _, T, C_w, H, W = weather_seq.shape

        # 1. Process Static Mosaic with Clay
        # Ensure mosaic is on the correct device
        cloudless_mosaic = cloudless_mosaic.to(weather_seq.device)

        # Pass mosaic through Clay encoder
        # Note: Clay outputs features at a reduced resolution (H', W')
        static_clay_feats = self.clay_encoder(cloudless_mosaic) # (B, C_clay_embed, H', W')

        # Project Clay features
        static_proj_feats = self.static_proj(static_clay_feats) # (B, proj_ch, H', W')

        # Resize projected features to match the target dynamic grid dimensions (H, W)
        static_resized = F.interpolate(static_proj_feats, size=(H, W), mode='bilinear', align_corners=False)

        # Generate initial hidden state for ConvGRU
        h = self.init_map(static_resized) # (B, hid_ch, H, W)

        # 2. Process Dynamic Features (Time sequence)
        dynamic_inputs = torch.cat([weather_seq, time_emb_seq, lst_seq], dim=2) # Shape: (B, T, C_dynamic, H, W)

        # Iterate through the time sequence (currently T=1)
        for t in range(T):
            x_t = dynamic_inputs[:, t, :, :, :] # Get input for time step t (B, C_dynamic, H, W)
            h = self.gru(x_t, h) # Update hidden state
            # Note: For T > 1, we would store intermediate hidden states if needed

        # 3. Predict UHI
        # Use the final hidden state to predict the UHI grid
        pred = self.regressor(h) # (B, 1, H, W)

        return pred.squeeze(1) # Remove channel dim -> (B, H, W) 