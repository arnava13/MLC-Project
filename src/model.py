import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import yaml
from box import Box
import numpy as np
from torchvision.transforms import v2
import pandas as pd # Added for Timestamp check

from src.Clay.src.module import ClayMAEModule

# -----------------------------------------------------------------------------
# 1. CLAY FEATURE EXTRACTOR ----------------------------------------------------
# Use to extract fixed spatial features for each city.
# -----------------------------------------------------------------------------

class ClayFeatureExtractor(nn.Module):
    """
    Loads a pre-trained Clay model from a local checkpoint and extracts features.
    Uses the encoder part of the model to get embeddings from a Sentinel-2 mosaic.
    Returns spatial patch embeddings.
    """

    def __init__(self, checkpoint_path: str, metadata_path: str, model_size: str = "large", bands: list = ["blue", "green", "red", "nir"], platform: str = "sentinel-2-l2a", gsd: int = 10):
        """
        Initializes the feature extractor.

        Args:
            checkpoint_path: Path to the local Clay model checkpoint (.ckpt).
            metadata_path: Path to the metadata.yaml file for normalization constants.
            model_size: Size of the Clay model (e.g., "base", "large"). Matches checkpoint.
            bands: List of band names in the input mosaic, matching metadata.yaml.
            platform: Platform name corresponding to the metadata (e.g., "sentinel-2-l2a").
            gsd: Ground sample distance of the input mosaic in meters.
        """
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        self.model_size = model_size
        self.bands = bands
        self.platform = platform
        self.gsd = gsd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = None # Will be set after model load
        self.embed_dim = None  # Will be set after model load

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Clay checkpoint not found at {self.checkpoint_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        # Load metadata
        self.metadata = Box(yaml.safe_load(open(self.metadata_path)))

        # Load the model from checkpoint
        self.model = ClayMAEModule.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device, # Explicitly load to target device
            model_size=self.model_size,
            metadata_path=str(self.metadata_path.resolve()),
            mask_ratio=0.0,
            shuffle=False,
        )
        self.model.eval()
        # self.model.to(self.device) # Already loaded to device with map_location
        print(f"Clay model loaded from {self.checkpoint_path} to {self.device}")

        # Store embed_dim and patch_size after loading
        self.embed_dim = self.model.model.encoder.dim
        self.patch_size = self.model.model.encoder.patch_size
        print(f"Clay model properties: embed_dim={self.embed_dim}, patch_size={self.patch_size}")

        # --- Define Target Input Size for Clay --- 
        self.target_input_size = (224, 224) # Standard ViT size

        # Prepare normalization based on metadata and selected bands
        self._prepare_normalization()

    def _prepare_normalization(self):
        """Prepares the normalization transform based on loaded metadata."""
        mean = []
        std = []
        waves = []
        platform_meta = self.metadata[self.platform]
        for band_name in self.bands:
            band_name_str = str(band_name)
            if band_name_str not in platform_meta.bands.mean:
                 raise ValueError(f"Band '{band_name_str}' not found in metadata for platform '{self.platform}'")
            mean.append(platform_meta.bands.mean[band_name_str])
            std.append(platform_meta.bands.std[band_name_str])
            waves.append(platform_meta.bands.wavelength[band_name_str])

        self.transform = v2.Compose([v2.Normalize(mean=mean, std=std)])
        self.waves = torch.tensor(waves, device=self.device)

        print(f"Normalization prepared for bands: {self.bands}")

    def _normalize_timestamp(self, date):
        if isinstance(date, (np.datetime64, str)):
            date_pd = pd.Timestamp(date)
        elif isinstance(date, pd.Timestamp):
            date_pd = date
        else:
            raise TypeError(f"Unsupported date type: {type(date)}")

        week = date_pd.isocalendar().week * 2 * np.pi / 52
        hour = date_pd.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def _normalize_latlon(self, lat, lon):
        lat_rad = lat * np.pi / 180
        lon_rad = lon * np.pi / 180
        return (math.sin(lat_rad), math.cos(lat_rad)), (math.sin(lon_rad), math.cos(lon_rad))

    def forward(self, sentinel_mosaic):
        """
        Extracts spatial features from a batch of Sentinel-2 mosaic tensors.
        Resizes input to a fixed size (e.g., 224x224) before feature extraction.

        Args:
            sentinel_mosaic (torch.Tensor): Input tensor (B, C, H, W).

        Returns:
            torch.Tensor: Extracted spatial features (B, D, H', W').
        """
        sentinel_mosaic = sentinel_mosaic.to(self.device)
        batch_size, C, H_orig, W_orig = sentinel_mosaic.shape

        # --- Resize input mosaic --- 
        # Ensure float for interpolate
        pixels_resized = F.interpolate(
            sentinel_mosaic.float(), 
            size=self.target_input_size, 
            mode='bilinear', 
            align_corners=False
        )
        B, C, H_resized, W_resized = pixels_resized.shape # Get resized dimensions

        # Normalize the *resized* pixels
        pixels_normalized = self.transform(pixels_resized) # Ensure float32 if transform expects it

        # --- Create placeholder metadata tensors (as required by Clay encoder) ---
        placeholder_date = pd.Timestamp("2021-07-15T12:00:00")
        time_norm = self._normalize_timestamp(placeholder_date)
        week_norm = [time_norm[0]] * batch_size
        hour_norm = [time_norm[1]] * batch_size
        time_tensor = torch.tensor(np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=self.device)

        placeholder_lat, placeholder_lon = 0.0, 0.0
        latlon_norm = self._normalize_latlon(placeholder_lat, placeholder_lon)
        lat_norm = [latlon_norm[0]] * batch_size
        lon_norm = [latlon_norm[1]] * batch_size
        latlon_tensor = torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=self.device)

        gsd_tensor = torch.full((batch_size,), self.gsd, device=self.device)
        # -----------------------------------------------------------------------

        # Adapt datacube structure for batch processing by Clay's encoder
        # Add a dummy time dimension (T=1) using the *resized* pixels
        pixels_unsqueezed = pixels_normalized.unsqueeze(1) # (B, 1, C, H_resized, W_resized)
        time_tensor_unsqueezed = time_tensor.unsqueeze(1)     # (B, 1, 4)
        latlon_tensor_unsqueezed = latlon_tensor.unsqueeze(1) # (B, 1, 4)
        gsd_tensor_unsqueezed = gsd_tensor.unsqueeze(1)       # (B, 1)

        spatial_embeddings_list = []
        with torch.no_grad():
            for i in range(batch_size):
                 single_datacube = {
                      "platform": self.platform,
                      "time": time_tensor_unsqueezed[i],     # (1, 4)
                      "latlon": latlon_tensor_unsqueezed[i], # (1, 4)
                      "pixels": pixels_unsqueezed[i],        # (1, C, H_resized, W_resized)
                      "gsd": gsd_tensor_unsqueezed[i],       # (1,)
                      "waves": self.waves,                   # (C,)
                 }
                 unmsk_patch, _, _, _ = self.model.model.encoder(single_datacube) # Output shape (1, N+1, D)
                 # print(f"[Debug ClayFeatureExtractor] unmsk_patch shape: {unmsk_patch.shape}") # Removed debug print

                 # Get spatial patch embeddings (excluding CLS token)
                 patch_embeddings = unmsk_patch[:, 1:, :] # Shape (1, N, D)

                 # Infer spatial dimensions (H', W') based on *resized* input
                 patch_size = self.patch_size
                 if isinstance(patch_size, tuple):
                     patch_size_h, patch_size_w = patch_size
                 else:
                     patch_size_h = patch_size_w = patch_size

                 # Calculate patches based on RESIZED dimensions
                 num_patches_h = H_resized // patch_size_h
                 num_patches_w = W_resized // patch_size_w
                 N = num_patches_h * num_patches_w

                 # Sanity checks
                 if patch_embeddings.shape[1] != N:
                      raise ValueError(f"Unexpected number of patches in Clay output. Expected {N} (from {H_resized}x{W_resized}), got {patch_embeddings.shape[1]}")
                 if patch_embeddings.shape[2] != self.embed_dim:
                      raise ValueError(f"Unexpected embedding dimension in Clay output. Expected {self.embed_dim}, got {patch_embeddings.shape[2]}")

                 # Reshape to spatial format: (1, N, D) -> (1, H', W', D) -> (1, D, H', W')
                 spatial_embedding = patch_embeddings.reshape(1, num_patches_h, num_patches_w, self.embed_dim)
                 spatial_embedding = spatial_embedding.permute(0, 3, 1, 2) # (1, D, H', W')
                 spatial_embeddings_list.append(spatial_embedding)

        # Stack spatial embeddings from the batch
        spatial_features = torch.cat(spatial_embeddings_list, dim=0) # Shape: (B, D, H', W')

        return spatial_features

# -----------------------------------------------------------------------------
# 2. CONV GRU CELL ------------------------------------------------------------
# -----------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Single-layer ConvGRU with configurable kernel size.""" # Docstring improved

    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x, h: (B, C, H, W) - Process single time step
        combined = torch.cat([x, h], dim=1)
        # Reset (r) and Update (z) gates
        z, r = torch.chunk(torch.sigmoid(self.conv_zr(combined)), 2, dim=1)
        # Candidate hidden state (h_tilde)
        combined_r = torch.cat([x, r * h], dim=1) # Use reset gate `r`
        h_tilde = torch.tanh(self.conv_h(combined_r))
        # Final hidden state (h_next) using update gate `z`
        h_next = (1 - z) * h + z * h_tilde
        return h_next

# Removed UHINetConvGRU class

# -----------------------------------------------------------------------------
# 3. MAIN UHI NET MODEL -------------------------------------------------------
# -----------------------------------------------------------------------------

class UHINet(nn.Module):
    """
    Main UHI prediction model. Designed for single time step processing,
    to be used within an external training loop that manages time steps and
    hidden states. Encodes static features (Clay, LST) once, and performs
    recurrent updates using dynamic features (weather, time) concatenated with
    static features at each step.
    """
    def __init__(self,
                 # Clay args
                 clay_checkpoint_path: str,
                 clay_metadata_path: str,
                 # Weather args
                 weather_channels: int, # Num channels in weather_seq input
                 # Time embedding args - now fixed at 2
                 time_embed_dim: int = 2,
                 # --- Args with defaults ---
                 proj_ch: int = 32, # Channels after projecting Clay features
                 clay_model_size: str = "large",
                 clay_bands: list = ["blue", "green", "red", "nir"], # Bands for Clay input mosaic
                 clay_platform: str = "sentinel-2-l2a",
                 clay_gsd: int = 10,
                 # LST args
                 lst_channels: int = 1, # Num channels in static LST map
                 use_lst: bool = True,
                 # ConvGRU args
                 gru_hidden_dim: int = 64, # Hidden dimension for ConvGRU cell
                 gru_kernel_size: int = 3,
    ):
        super().__init__()
        self.use_lst = use_lst
        self.proj_ch = proj_ch
        self.gru_hidden_dim = gru_hidden_dim
        self.weather_channels = weather_channels
        self.time_embed_dim = time_embed_dim # Should be 2 based on dataloader change
        self.lst_channels = lst_channels

        # --- Static Feature Extraction and Projection ---
        self.clay_backbone = ClayFeatureExtractor(
            checkpoint_path=clay_checkpoint_path,
            metadata_path=clay_metadata_path,
            model_size=clay_model_size,
            bands=clay_bands,
            platform=clay_platform,
            gsd=clay_gsd
        )
        # Freeze Clay backbone
        for param in self.clay_backbone.parameters():
            param.requires_grad = False

        clay_embed_dim = self.clay_backbone.embed_dim
        self.proj = nn.Conv2d(clay_embed_dim, self.proj_ch, kernel_size=1)

        # --- Calculate GRU Input Channels ---
        # Static features + Dynamic Features
        static_in_ch = self.proj_ch
        if self.use_lst:
            static_in_ch += self.lst_channels
        dynamic_in_ch = self.weather_channels + self.time_embed_dim
        gru_in_ch = static_in_ch + dynamic_in_ch

        # --- Recurrent Core ---
        self.gru = ConvGRUCell(in_ch=gru_in_ch, hid_ch=self.gru_hidden_dim, kernel_size=gru_kernel_size)

        # --- Prediction Head ---
        self.regressor = nn.Conv2d(self.gru_hidden_dim, 1, kernel_size=1)

        logging.info(f"UHINet initialized (Static features concatenated at each step):")
        logging.info(f"  Clay Embed Dim: {clay_embed_dim} -> Proj Dim: {self.proj_ch}")
        logging.info(f"  Use LST: {self.use_lst} (Channels: {self.lst_channels if self.use_lst else 0})")
        logging.info(f"  Static Feature Input Dim (Proj Clay [+ LST]): {static_in_ch}")
        logging.info(f"  Dynamic Feature Input Dim (Weather + Time): {dynamic_in_ch}")
        logging.info(f"  GRU Input Dim (Static + Dynamic): {gru_in_ch}")
        logging.info(f"  GRU Hidden Dim: {self.gru_hidden_dim}")


    def encode_and_project_static(self, sentinel_mosaic: torch.Tensor, static_lst: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encodes and projects static features (Clay + optional LST).
        Should be called once before the time loop in training.

        Args:
            sentinel_mosaic (torch.Tensor): Cloudless mosaic (B, C_clay, H, W).
            static_lst (torch.Tensor, optional): Static LST map (B, C_lst, H, W).
                                                Provide only if self.use_lst is True.

        Returns:
            torch.Tensor: Combined projected static features (B, proj_ch [+ lst_ch], H', W').
        """
        if self.use_lst and static_lst is None:
            raise ValueError("`static_lst` must be provided when `use_lst` is True.")
        if not self.use_lst and static_lst is not None:
            logging.warning("`static_lst` provided but `use_lst` is False. LST will be ignored.")

        # 1. Extract Clay features -> (B, D_clay, H', W')
        clay_features = self.clay_backbone(sentinel_mosaic)
        B, _, H_feat, W_feat = clay_features.shape

        # 2. Project Clay features -> (B, proj_ch, H', W')
        projected_clay = self.proj(clay_features)

        # 3. Combine with LST if used
        if self.use_lst:
            # Resize LST to match Clay feature map size (H', W')
            # Assume LST input is (B, C_lst, H, W) - needs resize if H,W != H_feat, W_feat
            if static_lst.shape[2:] != (H_feat, W_feat):
                 static_lst_resized = F.interpolate(static_lst, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            else:
                 static_lst_resized = static_lst

            combined_static = torch.cat([projected_clay, static_lst_resized], dim=1)
        else:
            combined_static = projected_clay # Shape (B, proj_ch, H', W')

        # Return combined projected static features
        # Shape: (B, proj_ch [+ lst_ch], H', W')
        return combined_static

    def step(self, x_t_combined: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Performs a single ConvGRU step.

        Args:
            x_t_combined (torch.Tensor): Combined input features for the current time step `t`.
                                       Shape: (B, static_feat_ch + dynamic_feat_ch, H', W').
                                       Includes static + weather + time features.
                                       Must match the spatial dimensions of h_prev.
            h_prev (torch.Tensor): Hidden state from the previous time step `t-1`.
                                  Shape: (B, gru_hidden_dim, H', W').

        Returns:
            torch.Tensor: Updated hidden state h_t (B, gru_hidden_dim, H', W').
        """
        # Check spatial dimensions match
        if x_t_combined.shape[2:] != h_prev.shape[2:]:
             raise ValueError(f"Spatial dimensions of x_t_combined ({x_t_combined.shape[2:]}) and "
                              f"h_prev ({h_prev.shape[2:]}) must match.")

        h_next = self.gru(x_t_combined, h_prev)
        return h_next

    def predict(self, h_final: torch.Tensor) -> torch.Tensor:
        """
        Applies the final regression head to the last hidden state.

        Args:
            h_final (torch.Tensor): The final hidden state after processing all time steps.
                                    Shape: (B, gru_hidden_dim, H', W').

        Returns:
            torch.Tensor: Predicted UHI map (B, 1, H', W').
        """
        pred = self.regressor(h_final)
        # Output shape: (B, 1, H', W') - Squeeze channel dim later if needed
        return pred


# Removed WeatherFeatureProcessor dummy class
