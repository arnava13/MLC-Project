## This file is sample code for feature extraction from satellite images.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import resnet18

class SatelliteFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()

        resnet = resnet18(pretrained=False)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # input conv
            *list(resnet.children())[1:-2]  # Drop the final FC layer
        )

        self.fpn = FeaturePyramidNetwork(in_channels_list=[64, 128, 256, 512],
                                         out_channels=out_channels)
        self.grid_pool = nn.AdaptiveAvgPool2d((33, 34))  # Down to roughly 1123 (33x34 ≒ 1123)

    def forward(self, x): # x: (B, 4, H, W)
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        x3 = self.backbone[2](x2)
        x4 = self.backbone[3](x3)
        x5 = self.backbone[4](x4)

        features = {
            '0': x1,
            '1': x2,
            '2': x3,
            '3': x4
        }

        fpn_outs = self.fpn(features)  # output: dict of feature maps

        fpn_feature = fpn_outs['0']  # shape: (B, C, H, W)
        pooled = self.grid_pool(fpn_feature)  # shape: (B, C, 33, 34)
        B, C, H, W = pooled.shape

        # reshape to per-grid
        out = pooled.view(B, C, -1).permute(0, 2, 1)  # → (B, 1123, C)
        return out
