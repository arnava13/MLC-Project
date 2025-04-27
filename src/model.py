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

from src.Clay.src.module import ClayMAEModule

# -----------------------------------------------------------------------------
# 1. CLAY FEATURE EXTRACTOR ----------------------------------------------------
# Use to extract fixed spatial features for each city.
# -----------------------------------------------------------------------------

class ClayFeatureExtractor(nn.Module):
    """
    Loads a pre-trained Clay model from a local checkpoint and extracts features.
    Uses the encoder part of the model to get embeddings from a Sentinel-2 mosaic.
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

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Clay checkpoint not found at {self.checkpoint_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        # Load metadata
        self.metadata = Box(yaml.safe_load(open(self.metadata_path)))

        # Load the model from checkpoint
        # Note: The tutorial uses additional args like dolls, mask_ratio etc.
        # These might be needed depending on the exact checkpoint and task.
        # For feature extraction (encoder only), mask_ratio=0.0 and shuffle=False seems appropriate.
        # We might need to adjust dolls/doll_weights if the checkpoint expects them.
        # Let's start minimal based on the tutorial's loading line.
        self.model = ClayMAEModule.load_from_checkpoint(
            self.checkpoint_path,
            # map_location=self.device, # load_from_checkpoint handles device placement
            # --- Arguments potentially needed based on checkpoint saving ---
            model_size=self.model_size,
            metadata_path=str(self.metadata_path.resolve()), # Pass path to module
            # dolls=[16, 32, 64, 128, 256, 768, 1024], # Example values, might need adjustment
            # doll_weights=[1, 1, 1, 1, 1, 1, 1],    # Example values
            mask_ratio=0.0, # Don't mask for feature extraction
            shuffle=False,  # Don't shuffle patches
            # --- End potential arguments ---
        )
        self.model.eval() # Set to evaluation mode
        self.model.to(self.device) # Ensure model is on the correct device
        print(f"Clay model loaded from {self.checkpoint_path} to {self.device}")

        # Prepare normalization based on metadata and selected bands
        self._prepare_normalization()

    def _prepare_normalization(self):
        """Prepares the normalization transform based on loaded metadata."""
        mean = []
        std = []
        waves = []
        platform_meta = self.metadata[self.platform]
        for band_name in self.bands:
            # Ensure band names are treated as strings for lookup
            band_name_str = str(band_name)
            if band_name_str not in platform_meta.bands.mean:
                 raise ValueError(f"Band '{band_name_str}' not found in metadata for platform '{self.platform}'")
            mean.append(platform_meta.bands.mean[band_name_str])
            std.append(platform_meta.bands.std[band_name_str])
            waves.append(platform_meta.bands.wavelength[band_name_str])

        self.transform = v2.Compose([v2.Normalize(mean=mean, std=std)])
        self.waves = torch.tensor(waves, device=self.device)

        print(f"Normalization prepared for bands: {self.bands}")
        # print(f"  Mean: {mean}")
        # print(f"  Std: {std}")
        # print(f"  Wavelengths: {waves}")


    # Helper functions for embeddings (from Clay tutorial)
    def _normalize_timestamp(self, date):
        # Using pandas Timestamp for isocalendar access
        date_pd = pd.Timestamp(date)
        week = date_pd.isocalendar().week * 2 * np.pi / 52
        hour = date_pd.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def _normalize_latlon(self, lat, lon):
        lat_rad = lat * np.pi / 180
        lon_rad = lon * np.pi / 180
        return (math.sin(lat_rad), math.cos(lat_rad)), (math.sin(lon_rad), math.cos(lon_rad))


    def forward(self, sentinel_mosaic):
        """
        Extracts features from a batch of Sentinel-2 mosaic tensors.

        Args:
            sentinel_mosaic (torch.Tensor): Input tensor representing the cloudless mosaic.
                                           Expected shape: (B, C, H, W), where C is the number of bands.

        Returns:
            torch.Tensor: Extracted features (class token embeddings). Shape: (B, embedding_dim)
        """
        sentinel_mosaic = sentinel_mosaic.to(self.device)
        batch_size = sentinel_mosaic.shape[0]

        # --- Prepare datacube inputs ---
        # 1. Pixels: Normalize the input mosaic
        # The mosaic is already (B, C, H, W). Need to normalize per batch item?
        # The tutorial normalizes the whole stack (T, C, H, W).
        # Let's apply normalization per item in the batch.
        pixels_normalized = self.transform(sentinel_mosaic.float()) # Ensure float32

        # 2. Time Embedding: Use a fixed placeholder date (e.g., middle of 2021)
        # The actual date isn't crucial for encoding a static mosaic, but required by the model structure.
        placeholder_date = pd.Timestamp("2021-07-15T12:00:00") # Example date
        time_norm = self._normalize_timestamp(placeholder_date)
        week_norm = [time_norm[0]] * batch_size
        hour_norm = [time_norm[1]] * batch_size
        time_tensor = torch.tensor(np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=self.device)

        # 3. Lat/Lon Embedding: Use placeholder coordinates (e.g., 0, 0)
        # Similar to time, exact location isn't primary for static feature extraction here.
        placeholder_lat, placeholder_lon = 0.0, 0.0
        latlon_norm = self._normalize_latlon(placeholder_lat, placeholder_lon)
        lat_norm = [latlon_norm[0]] * batch_size
        lon_norm = [latlon_norm[1]] * batch_size
        latlon_tensor = torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=self.device)

        # 4. GSD: Use the provided GSD
        gsd_tensor = torch.full((batch_size,), self.gsd, device=self.device)

        # 5. Wavelengths: Already prepared in _prepare_normalization
        waves_tensor = self.waves.unsqueeze(0).repeat(batch_size, 1) # Shape: (B, C)

        # --- Construct the datacube ---
        # Note: The Clay model might expect inputs without the batch dimension initially
        # if it was trained that way. The tutorial processes one stack (T, C, H, W).
        # We are processing a batch of static images (B, C, H, W).
        # Let's adapt the datacube structure assuming the model's encoder can handle batches.
        # If errors occur, we might need to loop through the batch.

        # Need to reshape pixels to (B, 1, C, H, W) or similar if model expects time dim?
        # The tutorial example runs encoder on (T, C, H, W). Our input is (B, C, H, W).
        # Let's assume the encoder forward might need adaptation or already handles B.
        # Trying with the current shape first.

        # The ClayMAEModule's forward pass takes the datacube.
        # The tutorial calls model.model.encoder directly. Let's try the module's forward.
        # If that fails, we'll call model.model.encoder.

        # Check model forward signature if possible, otherwise assume datacube dict.
        # Based on Clay source, ClayMAEModule.forward takes the datacube dict.

        datacube = {
            "platform": [self.platform] * batch_size, # Needs to be a list per batch item? Or just string? Check usage. Assume string OK.
            "time": time_tensor,        # (B, 4)
            "latlon": latlon_tensor,    # (B, 4)
            "pixels": pixels_normalized, # (B, C, H, W)
            "gsd": gsd_tensor,          # (B,)
            "waves": waves_tensor,      # (B, C)
            # "mask": None # Mask is handled internally based on mask_ratio? Set to None.
        }

        # --- Run the model encoder ---
        with torch.no_grad():
            # Using model.forward (ClayMAEModule)
            # output = self.model(datacube) # Returns dict with loss, embeddings etc.
            # We need the encoder output directly as per tutorial
            # output_dict = self.model.model.encoder(datacube) # This expects specific tensor shapes (T, C, H, W)?

            # Let's re-evaluate the tutorial call: `model.model.encoder(datacube)`
            # The datacube in the tutorial has pixels of shape (T, C, H, W).
            # Our pixels are (B, C, H, W).
            # We might need to unsqueeze a time dimension: (B, 1, C, H, W).
            # And adjust time/latlon/gsd/waves accordingly.

            # --- Attempt 2: Adapt datacube for time dimension ---
            pixels_unsqueezed = pixels_normalized.unsqueeze(1) # (B, 1, C, H, W)

            # Time/Latlon/GSD need to be (B, 1, *)
            time_tensor_unsqueezed = time_tensor.unsqueeze(1)     # (B, 1, 4)
            latlon_tensor_unsqueezed = latlon_tensor.unsqueeze(1) # (B, 1, 4)
            gsd_tensor_unsqueezed = gsd_tensor.unsqueeze(1)       # (B, 1)
            # Waves might need adjustment too? (B, C) -> (B, 1, C)? Assume (B, C) is fine.

            datacube_adapted = {
                "platform": self.platform, # Single string likely ok
                 # Use unsqueezed tensors
                "time": time_tensor_unsqueezed,
                "latlon": latlon_tensor_unsqueezed,
                "pixels": pixels_unsqueezed,
                "gsd": gsd_tensor_unsqueezed,
                "waves": self.waves, # Pass the base (C,) tensor, let model handle batching internally? Or repeat? Try base first.
                 # "waves": waves_tensor, # Or pass the repeated (B, C) tensor? Let's try repeated.
                # Pass the repeated waves tensor (B,C) - Model code likely handles this.
            }

            # Need to iterate over batch? The encoder likely expects T, C, H, W not B, T, C, H, W
            # Let's loop through the batch and call the encoder individually.

            embeddings_list = []
            for i in range(batch_size):
                 # Create datacube for a single batch item (T=1)
                 single_datacube = {
                      "platform": self.platform,
                      "time": time_tensor_unsqueezed[i],     # (1, 4)
                      "latlon": latlon_tensor_unsqueezed[i], # (1, 4)
                      "pixels": pixels_unsqueezed[i],        # (1, C, H, W)
                      "gsd": gsd_tensor_unsqueezed[i],       # (1,)
                      "waves": self.waves,                   # (C,) - Pass the original wavelengths tensor
                 }
                 # Call encoder for single item
                 unmsk_patch, unmsk_idx, msk_idx, msk_matrix = self.model.model.encoder(single_datacube)
                 # Extract class token embedding
                 cls_embedding = unmsk_patch[:, 0, :] # Shape: (1, embedding_dim)
                 embeddings_list.append(cls_embedding)

            # Stack embeddings from the batch
            embeddings = torch.cat(embeddings_list, dim=0) # Shape: (B, embedding_dim)


        # Expected output for UHI Net is (B, embed_dim, H', W') - spatial features
        # Clay's encoder output `unmsk_patch` includes patch embeddings + class token.
        # Shape is (num_unmasked_patches + 1, embedding_dim) for *each item* in the batch.
        # We extracted the class token `[:, 0, :]`. Shape (B, embedding_dim).
        # This is NOT spatial. The UHI Net expects spatial features from the backbone.

        # --- Revisit: Getting Spatial Features from Clay ---
        # We need the patch embeddings, not just the CLS token.
        # `unmsk_patch` contains embeddings for unmasked patches + CLS token.
        # Since mask_ratio=0.0, all patches are unmasked.
        # Shape of `unmsk_patch` for one item: (N+1, D), where N is num_patches, D is embed_dim.
        # We need to reshape `unmsk_patch[:, 1:, :]` (excluding CLS token) back into a spatial grid (B, D, H', W').

        # Get number of patches (H'*W') and embedding dim (D) from the model config or output.
        # embedding_dim = self.model.model.encoder.embed_dim # Accessing internal attribute
        # Assuming we know H', W' (e.g., input H/W divided by patch size)
        # patch_size = self.model.model.encoder.patch_size # e.g., 16
        # num_patches_h = H // patch_size
        # num_patches_w = W // patch_size

        # Let's re-run the loop and get the full `unmsk_patch`

        spatial_embeddings_list = []
        for i in range(batch_size):
             single_datacube = {
                  "platform": self.platform,
                  "time": time_tensor_unsqueezed[i],
                  "latlon": latlon_tensor_unsqueezed[i],
                  "pixels": pixels_unsqueezed[i],
                  "gsd": gsd_tensor_unsqueezed[i],
                  "waves": self.waves,
             }
             unmsk_patch, _, _, _ = self.model.model.encoder(single_datacube) # Shape (1, N+1, D)

             # Get spatial patch embeddings (excluding CLS token)
             patch_embeddings = unmsk_patch[:, 1:, :] # Shape (1, N, D)

             # Infer spatial dimensions (H', W') and embed_dim (D)
             # Need model's patch size and embed dim
             # Accessing internal attributes might be fragile. Let's assume standard ViT structure.
             embed_dim = self.model.model.embed_dim # Check actual attribute name in ClayMAEModule/Encoder
             patch_size = self.model.model.patch_embed.patch_size # Check actual attribute name
             # Handle potential tuple patch_size
             if isinstance(patch_size, tuple):
                 patch_size_h, patch_size_w = patch_size
             else:
                 patch_size_h = patch_size_w = patch_size

             _, _, H, W = sentinel_mosaic.shape
             num_patches_h = H // patch_size_h
             num_patches_w = W // patch_size_w
             N = num_patches_h * num_patches_w

             if patch_embeddings.shape[1] != N:
                  raise ValueError(f"Unexpected number of patches. Expected {N}, got {patch_embeddings.shape[1]}")
             if patch_embeddings.shape[2] != embed_dim:
                  raise ValueError(f"Unexpected embedding dimension. Expected {embed_dim}, got {patch_embeddings.shape[2]}")


             # Reshape to spatial format: (1, N, D) -> (1, H', W', D) -> (1, D, H', W')
             spatial_embedding = patch_embeddings.reshape(1, num_patches_h, num_patches_w, embed_dim)
             spatial_embedding = spatial_embedding.permute(0, 3, 1, 2) # (1, D, H', W')
             spatial_embeddings_list.append(spatial_embedding)

        # Stack spatial embeddings from the batch
        spatial_features = torch.cat(spatial_embeddings_list, dim=0) # Shape: (B, D, H', W')

        # print(f"Clay output features shape: {spatial_features.shape}")
        return spatial_features # Return spatial features


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


class UHINet(nn.Module):
    """
    Main UHI prediction model. Combines features from Clay (Sentinel),
    weather data, LST (optional), and time embeddings, then uses a ConvGRU
    followed by a projection head.
    """
    # def __init__(self, uhi_net_params, time_embed_dim, weather_channels, clay_model_name="google/clay-base-hf", clay_cache_dir=None, use_lst=True):
    def __init__(self,
                 # Clay args
                 clay_checkpoint_path: str,
                 clay_metadata_path: str,
                 # Weather args
                 weather_channels: int, # Num channels in weather_seq input
                 # Time embedding args
                 time_embed_dim: int, # Dimension of time_emb_seq input
                 # --- Args with defaults ---
                 proj_ch: int = 32, # Add proj_ch for UHINetConvGRU
                 clay_model_size: str = "large",
                 clay_bands: list = ["blue", "green", "red", "nir"], # Bands for Clay input mosaic
                 clay_platform: str = "sentinel-2-l2a",
                 clay_gsd: int = 10,
                 # LST args
                 lst_channels: int = 1, # Num channels in lst_seq input
                 use_lst: bool = True,
                 # ConvGRU args
                 gru_hidden_dim: int = 64, # Hidden dimension for ConvGRU cell
                 gru_kernel_size: int = 3,
    ):
        super().__init__()
        self.use_lst = use_lst

        self.clay_backbone = ClayFeatureExtractor(
            checkpoint_path=clay_checkpoint_path,
            metadata_path=clay_metadata_path,
            model_size=clay_model_size,
            bands=clay_bands,
            platform=clay_platform,
            gsd=clay_gsd
        )
        # Determine Clay output embedding dimension dynamically
        # We need to access the embed_dim after the model is loaded.
        # This requires initializing it first or passing the dim explicitly.
        # Let's access it after init.
        clay_embed_dim = self.clay_backbone.model.model.embed_dim # Access internal attribute

        self.weather_processor = WeatherFeatureProcessor(weather_channels) # Assuming this exists/is simple

        # Total static feature channels going into the UHINetConvGRU
        static_feature_channels = clay_embed_dim
        if self.use_lst:
            static_feature_channels += lst_channels

        # Dynamic feature channels (weather + time)
        dynamic_feature_channels = weather_channels + time_embed_dim # WeatherProcessor output channels + time_embed

        self.uhi_conv_gru = UHINetConvGRU(
            # Pass relevant arguments explicitly based on reverted UHINetConvGRU:
            sat_channels=static_feature_channels, # Map static features here (naming is from reverted code)
            weather_channels=weather_channels,    # Pass weather_channels
            proj_ch=proj_ch,                      # Pass proj_ch
            hid_ch=gru_hidden_dim                 # Pass gru_hidden_dim as hid_ch
        )

    def forward(self, sentinel_mosaic, weather_seq, lst_seq, time_emb_seq):
        """
        Args:
            sentinel_mosaic (torch.Tensor): Cloudless mosaic (B, C, H, W).
            weather_seq (torch.Tensor): Weather sequence (B, T, C_weather, H, W).
            lst_seq (torch.Tensor): LST sequence (B, T, C_lst, H, W). T=1 for static LST.
            time_emb_seq (torch.Tensor): Time embedding sequence (B, T, C_time_emb). Needs spatial broadcast.

        Returns:
            torch.Tensor: Predicted UHI map (B, H_out, W_out).
        """
        # 1. Extract static features from Clay
        # Output shape: (B, D, H', W')
        static_clay_features = self.clay_backbone(sentinel_mosaic)
        # print(f"Clay features shape: {static_clay_features.shape}")

        # 2. Prepare static LST (if used)
        static_lst = None
        if self.use_lst:
            if lst_seq is None or lst_seq.shape[1] == 0:
                 raise ValueError("use_lst is True, but lst_seq is None or empty.")
            # Extract the single static LST map (T=1)
            static_lst = lst_seq[:, 0, :, :, :] # Shape: (B, C_lst, H, W)
            # We might need to resize LST to match Clay feature map size (H', W')
            # This requires knowing H', W' from Clay. Let's assume UHINetConvGRU handles resizing/alignment.
            # print(f"Static LST shape: {static_lst.shape}")


        # 3. Process dynamic weather features (per step) - done inside ConvGRU now
        # weather_features_seq = self.weather_processor(weather_seq) # If processor is needed outside GRU

        # 4. Prepare dynamic time embeddings (per step) - needs spatial broadcast
        B, T, C_time_emb = time_emb_seq.shape
        # Assume target spatial size matches static_clay_features
        _, _, H_feat, W_feat = static_clay_features.shape
        time_emb_seq_spatial = time_emb_seq.unsqueeze(-1).unsqueeze(-1).expand(B, T, C_time_emb, H_feat, W_feat)
        # print(f"Time Embeddings shape (spatial): {time_emb_seq_spatial.shape}")
        # print(f"Weather Seq shape: {weather_seq.shape}")


        # 5. Pass features to ConvGRU
        # UHINetConvGRU now expects static features separately
        output = self.uhi_conv_gru(
            static_clay_features=static_clay_features,
            static_lst=static_lst, # Pass possibly None LST tensor
            dynamic_weather_seq=weather_seq, # Pass raw weather seq, GRU handles processing/concatenation
            dynamic_time_emb_seq=time_emb_seq_spatial # Pass spatially broadcasted time embeddings
        )

        # print(f"Final output shape: {output.shape}")
        return output


# Dummy WeatherFeatureProcessor if needed
class WeatherFeatureProcessor(nn.Module):
     def __init__(self, weather_channels):
          super().__init__()
          # Example: Just a passthrough or a simple conv
          self.conv = nn.Conv2d(weather_channels, weather_channels, kernel_size=1)

     def forward(self, weather_seq):
          # Input: (B, T, C, H, W)
          B, T, C, H, W = weather_seq.shape
          weather_seq_reshaped = weather_seq.view(B * T, C, H, W)
          processed = self.conv(weather_seq_reshaped)
          return processed.view(B, T, C, H, W) # Reshape back
