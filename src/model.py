import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. CLAY FEATURE EXTRACTOR ----------------------------------------------------
# Use to extract fixed spatial features for each city.
# -----------------------------------------------------------------------------

class ClayFeatureExtractor(nn.Module):
    """Wrap the Clay foundation model and return a *spatial* feature map (B,C,H',W').
    """

    def __init__(self, in_chans: int, ckpt: Optional[str] = None):
        super().__init__()
        self.in_chans = in_chans

        # Prefer official Clay loader (torch.hub) – keeps ~90 M pretrained params
        self.vit = torch.hub.load(
            "Clay-foundation/model", "clay_mae_base", source="github", pretrained=True
        )
        if in_chans != 4:  # Clay is trained on 4‑channel S‑2 chips (RGB‑NIR)
            self.vit.patch_embed.proj = nn.Conv2d(
                in_chans,
                self.vit.embed_dim,
                kernel_size=self.vit.patch_embed.patch_size,
                stride=self.vit.patch_embed.patch_size,
            )
    
        # Disable the classifier head – we only need features
        self.vit.reset_classifier(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W) → feat: (B,C_e,H',W')"""
        B = x.size(0)
        tokens = self.vit.patch_embed(x)  # (B, N, C_e)
        # +1 if model keeps CLS; locate it via attribute
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            tokens = tokens[:, 1:, :]  # drop CLS
        H = int(math.sqrt(tokens.size(1)))  # assume square grid of patches
        feat = tokens.transpose(1, 2).contiguous().view(B, -1, H, H)
        return feat  # (B, 768, H', W')


class SpatialProjector(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 3, p: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(num_inputs, num_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(self.relu(self.conv(x)))
    

# -----------------------------------------------------------------------------
# 2. PARAMETER-EFFICIENT RECURRENT CNN FOR SMALL DATA -------------------------
# -----------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Single-layer ConvGRU with 3×3 kernels."""

    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x, h: (B, C, H, W)
        combined = torch.cat([x, h], dim=1)
        z, r = torch.chunk(torch.sigmoid(self.conv_zr(combined)), 2, dim=1)
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class UHINetConvGRU(nn.Module):
    """Spatio-temporal model leveraging Clay features and spatial weather grids.

    Args:
        sat_channels:   # spectral bands in satellite mosaic
        weather_channels: # channels in weather grid (e.g., 3: max/min/precip)
        proj_ch:        # channels for projected Clay features
        hid_ch:         # hidden channels in ConvGRU
    """

    def __init__(
        self,
        sat_channels: int,
        weather_channels: int = 3,
        proj_ch: int = 32,
        hid_ch: int = 32,
    ):
        super().__init__()
        # Clay encoder
        self.encoder = ClayFeatureExtractor(sat_channels)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.proj = nn.Conv2d(768, proj_ch, kernel_size=1)

        # ConvGRU: input weather grid channels -> hidden
        self.gru = ConvGRUCell(weather_channels, hid_ch)

        self.regressor = nn.Conv2d(hid_ch, 1, kernel_size=1)

        self.init_map = (
            nn.Conv2d(proj_ch, hid_ch, kernel_size=1) if proj_ch != hid_ch else nn.Identity()
        )

    def forward(self, sat_img: torch.Tensor, weather_seq: torch.Tensor) -> torch.Tensor:
        """sat_img: (B,C,H,W), weather_seq: (B,T,C_w,H,W)"""
        B, C, H, W = sat_img.shape

        # Initial spatial hidden state
        feat = self.encoder(sat_img)
        feat = F.interpolate(self.proj(feat), size=(H, W), mode="bilinear", align_corners=False)
        h = self.init_map(feat)

        T = weather_seq.size(1)
        for t in range(T):
            x_t = weather_seq[:, t]  # (B,C_w,H,W)
            h = self.gru(x_t, h)

        pred = self.regressor(h)
        return pred.squeeze(1)
