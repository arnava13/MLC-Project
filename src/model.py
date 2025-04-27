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

    If Clay is not available locally the wrapper falls back to a timm ViT‑MAE with
    matching patch‑size so that the rest of the pipeline still runs.  In either
    case we re‑shape the patch tokens back to 2‑D and *drop* the CLS token.
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
# 2. TEMPORAL CONVOLUTIONAL NETWORK (TCN) BUILDING BLOCKS ----------------------
# -----------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """A single TCN layer with dilated causal conv, followed by ReLU + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # ensure residual connection has matching channels
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------------------------------------------------------
# 3. UHI REGRESSOR WITH SPATIAL PROJECTOR + TCN -------------------------------
# -----------------------------------------------------------------------------

class SimpleTemporalBlock(nn.Module):
    """A lightweight causal block without padding chomp – suitable for small data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        return self.relu(out + self.downsample(x))


class SimpleTCN(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i  # still use expanding receptive field
            layers.append(SimpleTemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UHINet(nn.Module):
    """Spatio-temporal network for UHI regression using Clay/ViT features.

    Pipeline:
        1. Clay (or ViT) encoder → spatial feature map (low-res)
        2. SpatialProjector (conv) → project to desired channel dim
        3. Upsample to original input grid size
        4. Flatten spatial grid → SimpleTCN operates on each grid-cell sequence
        5. Linear regression head → predict UHI per grid cell
    """

    def __init__(
        self,
        in_chans: int,
        proj_channels: int = 32,  # Reduced channels for smaller dataset
        tcn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,  # Reduced dropout for smaller dataset
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 32]

        # 1. Spatial encoder (Clay or ViT)
        self.encoder = ClayFeatureExtractor(in_chans)
        # Clay/ViT outputs 768 channels
        encoder_output_dim = 768

        # 2. Spatial projector to reduce channel dimension
        self.spatial_proj = SpatialProjector(num_inputs=encoder_output_dim, num_channels=proj_channels, kernel_size=1, p=dropout) # Use 1x1 conv for projection

        # 4. Temporal ConvNet to model per-pixel time series
        self.tcn = SimpleTCN(num_inputs=proj_channels, num_channels=tcn_channels, kernel_size=kernel_size, dropout=dropout)

        # 5. Regress final hidden features to scalar UHI value
        self.regressor = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → preds: (B, H, W)"""
        B, T, C, H, W = x.shape

        # 1. Spatial encoding for each timestamp
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        feat_enc = self.encoder(x)  # (B*T, 768, h_enc, w_enc)
        _, _, h_enc, w_enc = feat_enc.shape

        # 2. Project channels
        feat_proj = self.spatial_proj(feat_enc) # (B*T, P, h_enc, w_enc)
        P = feat_proj.size(1)

        # 3. Upsample to original grid size
        if (h_enc, w_enc) != (H, W):
             feat_upsampled = F.interpolate(feat_proj, size=(H, W), mode="bilinear", align_corners=False) # (B*T, P, H, W)
        else:
             feat_upsampled = feat_proj # (B*T, P, H, W)

        # Reshape back to time dimension
        feat = feat_upsampled.view(B, T, P, H, W)

        # 4. Prepare for TCN: gather per-pixel sequences (B*H*W, P, T)
        feat = feat.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, P, T)
        feat = feat.view(B * H * W, P, T)

        # Temporal modelling
        out = self.tcn(feat)  # (B*H*W, hidden, T)
        out_last = out[:, :, -1]  # (B*H*W, hidden)

        # 5. Regression
        preds = self.regressor(out_last)  # (B*H*W, 1)
        preds = preds.view(B, H, W)
        return preds
