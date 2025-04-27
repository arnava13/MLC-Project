import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. CLAY FEATURE EXTRACTOR ----------------------------------------------------
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

        try:
            # Prefer official Clay loader (torch.hub) – keeps ~90 M pretrained params
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
        except Exception:
            import timm

            self.vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, in_chans=in_chans
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

# -----------------------------------------------------------------------------
# 2. SIMPLE TEMPORAL CONVOLUTIONAL NETWORK ------------------------------------
# -----------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, p: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.chomp = lambda t: t[:, :, :-padding] if padding else t
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 3, p: float = 0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2 ** i, p))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C, T)
        return self.network(x)

# -----------------------------------------------------------------------------
# 3. UHI REGRESSOR WITH CONV + TCN HEAD --------------------------------------
# -----------------------------------------------------------------------------

class UHINet(nn.Module):
    """Full model composed of
       1) Clay (or fallback ViT) feature extractor – spatial encoder
       2) 3‑D Conv to fuse space + time
       3) TemporalConvNet to model diurnal dynamics

       Expects a batch of *T* satellite frames stacked along the *time* dim
       -> dataloader must collate (satellite, weather, meta) into lists and stack.
    """

    def __init__(self, n_bands: int, T: int = 12, use_lst: bool = True):
        super().__init__()
        in_ch = n_bands + int(use_lst)
        self.T = T
        self.encoder = ClayFeatureExtractor(in_ch)
        self.enc_out_channels = 768  # vit_base hidden dim

        # 3‑D conv over (T, H', W') – keep kernel = 3 in each dim
        self.conv3d = nn.Conv3d(
            self.enc_out_channels,
            256,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )

        # After spatial fusion, collapse H'×W' with GAP
        # -> tensor shape (B, 256, T)
        self.tcn = TemporalConvNet(256, [256, 128, 64])
        self.head = nn.Sequential(nn.Conv1d(64, 1, 1))  # (B,1,T)

    def forward(self, sat_batch: torch.Tensor, weather: torch.Tensor, time_feats: torch.Tensor):
        """sat_batch : (B,T,C,H,W)  – stack of hourly chips
           weather   : (B,T,3)       – broadcast later if desired
           time_feats: (B,T,2) sin‑cos hour
        """
        B, T, C, H, W = sat_batch.shape
        assert T == self.T, "Time dimension mismatch"

        # Merge batch & time to feed encoder efficiently
        sat_flat = sat_batch.view(B * T, C, H, W)
        f = self.encoder(sat_flat)               # (B*T, C_e, H', W')
        C_e, H_p, W_p = f.shape[1:]
        f = f.view(B, T, C_e, H_p, W_p).permute(0, 2, 1, 3, 4)  # (B, C_e, T, H', W')

        x = self.conv3d(f)                      # (B, 256, T, H', W')
        x = F.adaptive_avg_pool3d(x, (self.T, 1, 1)).squeeze(-1).squeeze(-1)  # (B,256,T)

        x = self.tcn(x)                         # (B, 64, T)
        out = self.head(x).squeeze(1)           # (B, T)
        return out

