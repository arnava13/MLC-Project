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
# Pretrained Feature Extractors -----------------------------------------------
# -----------------------------------------------------------------------------

class ClayFeatureExtractor(nn.Module):
    """
    Loads a pre-trained Clay model from a local checkpoint and extracts features.
    Uses the encoder part of the model to get embeddings from a Sentinel-2 mosaic.
    Returns spatial patch embeddings.
    If freeze_backbone is False, only the final identified projection layer ('proj') of the encoder is unfrozen.
    If freeze_backbone is True, the entire model remains frozen.
    """

    def __init__(self, checkpoint_path: str, metadata_path: str, model_size: str = "large", bands: list = ["blue", "green", "red", "nir"], platform: str = "sentinel-2-l2a", gsd: int = 10, freeze_backbone: bool = True):
        """
        Initializes the feature extractor.

        Args:
            checkpoint_path: Path to the local Clay model checkpoint (.ckpt).
            metadata_path: Path to the metadata.yaml file for normalization constants.
            model_size: Size of the Clay model (e.g., "base", "large"). Matches checkpoint.
            bands: List of band names in the input mosaic, matching metadata.yaml.
            platform: Platform name corresponding to the metadata (e.g., "sentinel-2-l2a").
            gsd: Ground sample distance of the input mosaic in meters.
            freeze_backbone (bool): If True, freezes the entire Clay backbone weights.
                                    If False, only the final identified projection layer ('proj') is trainable.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        # self.model_size = model_size # Store for reference, but don't pass to init directly
        self.bands = bands
        self.platform = platform
        self.gsd = gsd
        # Store the flag, but apply freezing logic below
        # self.freeze_backbone = freeze_backbone # Original line removed

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Clay checkpoint not found at {self.checkpoint_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Clay metadata not found at {self.metadata_path}")

        # Load metadata (Box allows attribute-style access)
        self.metadata_config = Box(yaml.safe_load(open(self.metadata_path)))

        # --- Re-applying Manual Checkpoint Loading --- 
        print(f"Manually loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu') # Load to CPU first

        if "hyper_parameters" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'hyper_parameters' key.")
        if "state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'state_dict' key.")
            
        hparams = checkpoint["hyper_parameters"]
        state_dict = checkpoint["state_dict"]

        # Prepare arguments for ClayMAEModule constructor from hyperparameters
        model_args = hparams.copy() # Start with all hparams
        
        # Override metadata_path with the absolute one
        model_args["metadata_path"] = str(self.metadata_path.resolve())
        
        # Remove internal Lightning instantiator key
        model_args.pop("_instantiator", None)

        # Instantiate the model using arguments from the checkpoint
        print("Instantiating ClayMAEModule manually...")
        self.model = ClayMAEModule(**model_args)
        
        # Load the state dictionary
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'model.' prefix for loading into the nested model
            if k.startswith("model."):
                 name = k[len("model."):] 
                 adjusted_state_dict[name] = v
            # else: # Optionally handle keys that don't start with model. if needed
            #     adjusted_state_dict[k] = v 
        
        print("Loading state_dict manually into self.model.model...")
        # --- MODIFIED: Load into self.model.model --- 
        missing_keys, unexpected_keys = self.model.model.load_state_dict(adjusted_state_dict, strict=False)
        # -------------------------------------------
        if missing_keys:
            print(f"Warning: Missing keys in state_dict (relative to self.model.model): {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict (relative to self.model.model): {unexpected_keys}")

        self.model.to(self.device) # Move model to target device
        # Set to eval mode initially. Will be set to train mode later if needed.
        self.model.eval()
        # ----------------------------------------

        # --- MODIFIED Freezing Logic ---
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Identify the final projection layer within ClayMAE
        final_encoder_layer = None
        # Check for self.model.model.proj based on architecture printout
        if hasattr(self.model.model, 'proj') and isinstance(self.model.model.proj, nn.Linear):
             final_encoder_layer = self.model.model.proj
             logging.info("Identified final encoder layer as self.model.model.proj")
        # Removed previous checks for encoder.fc and encoder.head
        # elif hasattr(self.model.model, 'encoder') and hasattr(self.model.model.encoder, 'fc'):
        #     final_encoder_layer = self.model.model.encoder.fc
        #     logging.info("Identified final encoder layer as self.model.model.encoder.fc")
        else:
             logging.warning("Could not automatically identify final encoder layer ('self.model.model.proj'). Check Clay model structure.")
             

        if not freeze_backbone:
            if final_encoder_layer is not None:
                logging.info("Unfreezing the final encoder layer (self.model.model.proj) of the Clay backbone.")
                for param in final_encoder_layer.parameters():
                    param.requires_grad = True
                self.model.train() # Set model to train mode if part of it is unfrozen
            else:
                 logging.warning("freeze_backbone is False, but could not identify final encoder layer (proj) to unfreeze.")
                 self.model.eval() # Keep in eval if nothing was unfrozen
        else:
            logging.info("Keeping Clay backbone frozen.")
            self.model.eval() # Ensure model is in eval mode if fully frozen
        # --- END MODIFIED Freezing Logic ---

        # Get embed_dim and patch_size from hparams, inferring embed_dim from model_size
        # --- MODIFIED: Override patch_size based on observed error --- 
        try:
            # --- Set patch_size explicitly --- 
            self.patch_size = 16 # Override based on error: 196 patches from 224x224 -> 16x16 patches
            print(f"WARNING: Overriding patch size from hparams. Using fixed patch_size = {self.patch_size}")
            # --------------------------------

            # --- Get model_size and infer embed_dim from hparams --- 
            self.model_size = hparams.get('model_size')
            if self.model_size is None:
                 # Try to get teacher model name if model_size is missing
                 teacher_name = hparams.get('teacher')
                 if teacher_name and 'large' in teacher_name.lower():
                     self.model_size = 'large'
                     print("Inferred model_size='large' from teacher hyperparameter.")
                 else:
                     raise KeyError("'model_size' key not found and could not be inferred from 'teacher' key in checkpoint hyperparameters.")
                 
            if self.model_size == 'large':
                 self.embed_dim = 1024 # Standard for ViT-Large
            # Add elif for other sizes like 'base' if needed
            # elif self.model_size == 'base':
            #     self.embed_dim = 768 
            else:
                 # Fallback: Try to get embed_dim from the instantiated model if size is not 'large' or known
                 try:
                     vit_backbone = self.model.model.encoder # Or appropriate path
                     if hasattr(vit_backbone, 'embed_dim'):
                         self.embed_dim = vit_backbone.embed_dim
                         print(f"Inferred embed_dim={self.embed_dim} from model structure for model_size '{self.model_size}'.")
                     elif hasattr(vit_backbone, 'num_features'): # Common in timm
                         self.embed_dim = vit_backbone.num_features
                         print(f"Inferred embed_dim={self.embed_dim} (from num_features) for model_size '{self.model_size}'.")
                     else:
                         raise ValueError(f"Unsupported model_size '{self.model_size}' and could not infer embed_dim from model structure.")
                 except Exception as e:
                      raise ValueError(f"Unsupported model_size '{self.model_size}' found in hparams. Could not infer embed_dim: {e}")
            # --------------------------------------------------------

            print(f"Clay model properties: model_size={self.model_size}, embed_dim={self.embed_dim}, patch_size={self.patch_size} (patch_size OVERRIDDEN)")
            
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error determining model parameters from hparams/structure: {e}")
            raise ValueError("Failed to get required model parameters from checkpoint hyperparameters or model structure.") from e
        except Exception as e: # Catch other potential errors
             print(f"Unexpected error determining model parameters: {e}")
             raise

        # --- Define Target Input Size for Clay --- 
        self.target_input_size = (224, 224) # Standard ViT size

        # Prepare normalization based on metadata and selected bands
        self._prepare_normalization()

    def _prepare_normalization(self):
        """Prepares the normalization transform based on loaded metadata."""
        mean = []
        std = []
        waves = []
        # Use self.metadata_config now
        platform_meta = self.metadata_config[self.platform]
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

    def forward(self, sentinel_mosaic: torch.Tensor, 
                norm_time_tensor: torch.Tensor, 
                norm_latlon_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial features from a batch of Sentinel-2 mosaic tensors,
        using dynamic time and lat/lon information for each batch item.
        Resizes input mosaic to a fixed size before feature extraction.

        Args:
            sentinel_mosaic (torch.Tensor): Input tensor (B, C, H, W).
            norm_time_tensor (torch.Tensor): Normalized time tensor for the batch (B, 4).
                                             Expected format [sin(week), cos(week), sin(hour), cos(hour)].
            norm_latlon_tensor (torch.Tensor): Normalized lat/lon tensor for the batch (B, 4).
                                               Expected format [sin(lat), cos(lat), sin(lon), cos(lon)].

        Returns:
            torch.Tensor: Extracted spatial features (B, D, H', W').
        """
        sentinel_mosaic = sentinel_mosaic.to(self.device)
        norm_time_tensor = norm_time_tensor.to(self.device)
        norm_latlon_tensor = norm_latlon_tensor.to(self.device)
        batch_size, C, H_orig, W_orig = sentinel_mosaic.shape

        if norm_time_tensor.shape != (batch_size, 4):
            raise ValueError(f"Unexpected shape for norm_time_tensor: {norm_time_tensor.shape}. Expected ({batch_size}, 4)")
        if norm_latlon_tensor.shape != (batch_size, 4):
            raise ValueError(f"Unexpected shape for norm_latlon_tensor: {norm_latlon_tensor.shape}. Expected ({batch_size}, 4)")


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

        # --- Prepare dynamic metadata tensors (unsqueezing for T=1 dim) ---
        # Use the provided normalized tensors, removing placeholder logic
        time_tensor_unsqueezed = norm_time_tensor.unsqueeze(1)     # (B, 1, 4)
        latlon_tensor_unsqueezed = norm_latlon_tensor.unsqueeze(1) # (B, 1, 4)
        
        # GSD remains fixed based on initialization
        gsd_tensor = torch.full((batch_size,), self.gsd, device=self.device)
        gsd_tensor_unsqueezed = gsd_tensor.unsqueeze(1)       # (B, 1)
        # -----------------------------------------------------------------------

        # Adapt datacube structure for batch processing by Clay's encoder
        # Add a dummy time dimension (T=1) using the *resized* pixels
        pixels_unsqueezed = pixels_normalized.unsqueeze(1) # (B, 1, C, H_resized, W_resized)

        spatial_embeddings_list = []
        # --- MODIFIED: Use model's training state set in __init__ ---
        # context_manager = torch.no_grad() if self.freeze_backbone else torch.enable_grad() # Original removed
        # No context manager needed, rely on parameter requires_grad status and model mode
        # Ensure model is in the correct mode (eval or train) set in __init__
        # self.model.train(not self.freeze_backbone) # Explicitly set mode based on original flag
        with torch.set_grad_enabled(self.model.training): # Use model's current training state
            for i in range(batch_size):
                 single_datacube = {
                      "platform": self.platform,
                      "time": time_tensor_unsqueezed[i],     # (1, 4) - Now dynamic per sample
                      "latlon": latlon_tensor_unsqueezed[i], # (1, 4) - Now dynamic per sample
                      "pixels": pixels_unsqueezed[i],        # (1, C, H_resized, W_resized)
                      "gsd": gsd_tensor_unsqueezed[i],       # (1,)   - Still fixed
                      "waves": self.waves,                   # (C,)   - Still fixed
                 }
                 # --- Encoder forward pass --- 
                 unmsk_patch, _, _, _ = self.model.model.encoder(single_datacube) # Output shape (1, N+1, D)
                 # --------------------------

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
# NEW: High-Resolution Elevation Branch ---------------------------------------
# -----------------------------------------------------------------------------

class HighResElevationBranch(nn.Module):
    """
    A convolutional branch to process high-resolution elevation data (DEM/DSM)
    and downsample it to match the feature map resolution of the main backbone.
    Uses strided convolutions for downsampling.
    """
    def __init__(self, in_channels: int = 1,
                 start_channels: int = 16,
                 out_channels: int = 32,
                 num_downsample_layers: int = 3, # Adjust based on resolution diff
                 kernel_size: int = 3):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        padding = kernel_size // 2
        
        # Initial convolution
        layers.append(nn.Conv2d(current_channels, start_channels, kernel_size=kernel_size, padding=padding, stride=1))
        layers.append(nn.BatchNorm2d(start_channels))
        layers.append(nn.ReLU(inplace=True))
        current_channels = start_channels
        
        # Downsampling layers with stride=2
        for i in range(num_downsample_layers):
            next_channels = current_channels * 2
            layers.append(nn.Conv2d(current_channels, next_channels,
                                    kernel_size=kernel_size, padding=padding, stride=2))
            layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels
            
        # Final convolution to get desired output channels (without further downsampling)
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1, stride=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.branch = nn.Sequential(*layers)

        logging.info(f"HighResElevationBranch initialized: In={in_channels}, Start={start_channels}, Out={out_channels}, DownsampleLayers={num_downsample_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)

# -----------------------------------------------------------------------------
# CNN HEADS -------------------------------------------------------------------
# -----------------------------------------------------------------------------

# --- Added SimpleCNNHead back ---
class SimpleCNNHead(nn.Module):
    """Simple multi-layer CNN head with optional dropout."""
    def __init__(self, in_channels, hidden_dims, kernel_size=3, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
        cnn_layers = []
        current_channels = in_channels
        padding = kernel_size // 2
        for i, hidden_dim in enumerate(hidden_dims):
            cnn_layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=kernel_size, padding=padding))
            cnn_layers.append(nn.BatchNorm2d(hidden_dim))
            cnn_layers.append(nn.ReLU(inplace=True))
            current_channels = hidden_dim
        self.cnn_core = nn.Sequential(*cnn_layers)
        self.final_bn = nn.BatchNorm2d(current_channels)
        self.regressor = nn.Conv2d(current_channels, 1, kernel_size=1)

        # Log dimensions
        logging.info(f"SimpleCNNHead initialized: Input Ch={in_channels}, Hidden Dims={hidden_dims}, Kernel={kernel_size}, Dropout={dropout}")
        logging.info(f"  Output Dim (before final BN/Regressor): {current_channels}")

    def forward(self, x):
        x = self.cnn_core(x)
        x = self.final_bn(x)
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.regressor(x)
        return x

# --- UNet-style Blocks ---
class UNetConvBlock(nn.Module):
    """Helper: Conv(3x3, padding=1) -> BN -> ReLU -> Conv(3x3, padding=1) -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNetUpBlock(nn.Module):
    """Helper: Upsample(ConvTranspose2d 2x2, stride=2) -> Concat -> UNetConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels is channels from previous layer, out_channels is for the ConvBlock
        # The actual input to ConvBlock will be in_channels//2 (from upsampling) + in_channels//2 (from skip connection) = in_channels
        # The output from ConvBlock will be out_channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetConvBlock(in_channels, out_channels) # Takes combined channels

    def forward(self, x1, x2):
        """
        x1: input from previous layer (to be upsampled)
        x2: input from corresponding skip connection
        """
        x1 = self.up(x1)
        # Input tensors to ConvTranspose2d must have the same spatial size.
        # Pad x1 if necessary to match x2's spatial dimensions after upsampling
        # Calculate difference: target_size - current_size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # Concatenate along channel dimension
        return self.conv(x)

class UNetStyleHead(nn.Module):
    """
    U-Net style head with downsampling, upsampling, and skip connections.
    Uses ConvTranspose2d for upsampling.
    Takes combined Clay, weather, and optional LST features.
    Depth is configurable.
    """
    def __init__(self, in_channels: int, base_channels: int = 64, depth: int = 4):
        super().__init__()
        if depth < 1:
            raise ValueError("UNet depth must be at least 1")
        self.depth = depth
        features = base_channels

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Initial Convolution
        self.inc = UNetConvBlock(in_channels, features)

        # Downsampling Path
        current_channels = features
        for i in range(depth):
            self.downs.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    UNetConvBlock(current_channels, current_channels * 2)
                )
            )
            current_channels *= 2
        # Bottleneck is implicitly the last downsampling block's output
        self.bottleneck_channels = current_channels

        # Upsampling Path
        # Starts from bottleneck channels, goes up to base_channels
        for i in range(depth):
            # Input to UpBlock: current_channels (from below), Output: current_channels // 2
            # Skip connection comes from layer with current_channels // 2
            self.ups.append(
                UNetUpBlock(current_channels, current_channels // 2)
            )
            current_channels //= 2

        # Final 1x1 convolution
        self.outc = nn.Conv2d(features, 1, kernel_size=1)

        logging.info(f"UNetStyleHead initialized: Input Ch={in_channels}, Base Ch={base_channels}, Depth={depth}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the U-Net head.

        Args:
            x (torch.Tensor): Combined projected input features
                              (ProjClay+Weather[+LST])
                              Shape (B, C_in, H_feat, W_feat)

        Returns:
            torch.Tensor: Predicted UHI map at feature resolution (B, 1, H_out, W_out)
        """
        # --- Encoder ---
        skip_connections = []
        out = self.inc(x) # Initial Block -> base_channels
        skip_connections.append(out)

        for i in range(self.depth):
            out = self.downs[i](out)
            if i < self.depth - 1: # Don't store bottleneck output as skip
                skip_connections.append(out)

        # Bottleneck output is `out` after the loop

        # --- Decoder ---
        # Iterate through upsampling blocks and corresponding skip connections in reverse
        for i in range(self.depth):
            skip = skip_connections.pop() # Get corresponding skip connection
            out = self.ups[i](out, skip)

        # --- Final Output ---
        logits = self.outc(out) # (B, 1, H_up, W_up)

        # Output is at the resolution determined by U-Net structure
        return logits

# -----------------------------------------------------------------------------
# 5. CNN-BASED UHI NET MODEL  ----------------------------
# -----------------------------------------------------------------------------

class UHINetCNN(nn.Module):
    """
    Feedforward CNN UHI prediction model using a U-Net style head.
    Uses dynamic Clay features, static LST (optional), and dynamic weather.
    Upsampling is handled by the UNetStyleHead using ConvTranspose2d.
    """
    def __init__(self,
                 # Clay args
                 clay_checkpoint_path: str,
                 clay_metadata_path: str,
                 weather_channels: int,
                 freeze_backbone: bool = True,
                 # --- Args with defaults ---
                 proj_ch: int = 32, # RESTORED - Channels after projecting Clay features (for SimpleCNNHead)
                 clay_model_size: str = "large",
                 clay_bands: list = ["blue", "green", "red", "nir"],
                 clay_platform: str = "sentinel-2-l2a",
                 clay_gsd: int = 10,
                 # LST args
                 lst_channels: int = 1,
                 use_lst: bool = True,
                 # --- Head Selection & Args ---
                 head_type: str = 'unet', # 'unet' or 'simple_cnn'
                 # SimpleCNN Head Args
                 cnn_hidden_dims: List[int] = [64, 32],
                 cnn_kernel_size: int = 3,
                 cnn_dropout: float = 0.0,
                 # UNet Head Args
                 unet_base_channels: int = 64, # Base channels for U-Net Head
                 unet_depth: int = 4,           # Depth of the U-Net
                 # High-Res Elevation Branch Args
                 include_dem_branch: bool = False,
                 elevation_out_channels: int = 32,
                 include_dsm_branch: bool = False
    ):
        super().__init__()
        self.use_lst = use_lst
        self.proj_ch = proj_ch # Restored for SimpleCNNHead
        self.weather_channels = weather_channels
        self.lst_channels = lst_channels
        self.head_type = head_type
        self.include_dem_branch = include_dem_branch
        self.include_dsm_branch = include_dsm_branch
        self.elevation_out_channels = elevation_out_channels

        # --- Dynamic Clay Feature Extraction (No projection layer here) ---
        self.clay_backbone = ClayFeatureExtractor(
            checkpoint_path=clay_checkpoint_path,
            metadata_path=clay_metadata_path,
            model_size=clay_model_size,
            bands=clay_bands,
            platform=clay_platform,
            gsd=clay_gsd,
            freeze_backbone=freeze_backbone # Pass flag here
        )

        clay_embed_dim = self.clay_backbone.embed_dim
        # --- RESTORED: Projection layer (used by *both* head types now) ---
        self.proj = nn.Conv2d(clay_embed_dim, self.proj_ch, kernel_size=1)

        # --- Instantiate Selected Head ---
        # Calculate Head Input Channels: Projected Clay + Weather [+ LST] [+ DEM] [+ DSM]
        head_in_ch = self.proj_ch + self.weather_channels
        if self.use_lst:
            head_in_ch += self.lst_channels
        if self.include_dem_branch:
            head_in_ch += self.elevation_out_channels
        if self.include_dsm_branch:
            head_in_ch += self.elevation_out_channels
        
        if self.head_type == 'unet':
            self.head = UNetStyleHead(
                in_channels=head_in_ch, # Takes projected features now
                base_channels=unet_base_channels,
                depth=unet_depth # Pass depth
            )
            logging.info(f"UHINetCNN initialized (Feedforward with UNetStyleHead):")
            logging.info(f"  Clay Backbone Frozen (except maybe last layer): {freeze_backbone}")
            logging.info(f"  Clay Embed Dim: {clay_embed_dim} -> Proj Dim: {self.proj_ch}")
            logging.info(f"  Use LST: {self.use_lst} (Channels: {self.lst_channels if self.use_lst else 0})")
            logging.info(f"  Weather Channels: {self.weather_channels}")
            logging.info(f"  UNet Head Input Channels (ProjClay+Weather[+LST]): {head_in_ch}")
            logging.info(f"  UNet Head Base Channels: {unet_base_channels}")
            logging.info(f"  UNet Head Depth: {unet_depth}")
            logging.info(f"  Include DEM Branch: {self.include_dem_branch}")
            logging.info(f"  Include DSM Branch: {self.include_dsm_branch}")

        elif self.head_type == 'simple_cnn':
            self.head = SimpleCNNHead(
                in_channels=head_in_ch,
                hidden_dims=cnn_hidden_dims,
                kernel_size=cnn_kernel_size,
                dropout=cnn_dropout
            )
            # Logging info for SimpleCNNHead is done within its __init__
            logging.info(f"UHINetCNN initialized (Feedforward with SimpleCNNHead):")
            logging.info(f"  Include DEM Branch: {self.include_dem_branch}")
            logging.info(f"  Include DSM Branch: {self.include_dsm_branch}")
            logging.info(f"  Clay Backbone Frozen (except maybe last layer): {freeze_backbone}")
            logging.info(f"  Clay Embed Dim: {clay_embed_dim} -> Proj Dim: {self.proj_ch}")
            logging.info(f"  Use LST: {self.use_lst} (Channels: {self.lst_channels if self.use_lst else 0})")
            logging.info(f"  Weather Channels: {self.weather_channels}")
            logging.info(f"  SimpleCNN Head Input Channels (ProjClay+Weather[+LST]): {head_in_ch}")
            # SimpleCNNHead logs its own details

        else:
            raise ValueError(f"Unknown head_type: {self.head_type}. Choose 'unet' or 'simple_cnn'.")

        # Projection layer is always trainable as it feeds into the selected head.
        for param in self.proj.parameters():
            param.requires_grad = True
        logging.info("  Projection layer (self.proj) set to trainable.")

    def forward(self, cloudless_mosaic: torch.Tensor,
                norm_time_tensor: torch.Tensor,
                norm_latlon_tensor: torch.Tensor,
                weather: torch.Tensor,
                target_h_w: Tuple[int, int], 
                static_lst: Optional[torch.Tensor] = None,
                high_res_dem: Optional[torch.Tensor] = None,
                high_res_dsm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the CNN model using dynamic Clay features
        and the selected head, incorporating optional high-resolution elevation features.

        Args:
            cloudless_mosaic (torch.Tensor): Cloudless mosaic (B, C_clay, H_orig, W_orig).
            norm_time_tensor (torch.Tensor): Normalized time tensor for Clay (B, 4).
            norm_latlon_tensor (torch.Tensor): Normalized lat/lon tensor for Clay (B, 4).
            weather (torch.Tensor): Weather data for the time step (B, 1, C_weather, H_orig, W_orig).
            target_h_w (Tuple[int, int]): The desired output height and width (H_orig, W_orig).
            static_lst (torch.Tensor, optional): Static LST map (B, 1, C_lst, H_low_res, W_low_res).
                                                Provide only if self.use_lst is True.
            high_res_dem (torch.Tensor, optional): High-resolution DEM map (B, 1, H_high_res, W_high_res).
                                                Provide only if self.include_dem_branch is True.
            high_res_dsm (torch.Tensor, optional): High-resolution DSM map (B, 1, H_high_res, W_high_res).
                                                Provide only if self.include_dsm_branch is True.

        Returns:
            torch.Tensor: Predicted UHI map at feature resolution (B, 1, H_out, W_out)
        """
        # 0. Input Checks and Preparation
        if self.use_lst and static_lst is None:
            raise ValueError("`static_lst` must be provided when `use_lst` is True.")
        if not self.use_lst and static_lst is not None:
            logging.warning("`static_lst` provided but `use_lst` is False. LST will be ignored.")

        # Check DEM/DSM inputs match config flags
        if self.include_dem_branch and high_res_dem is None:
            raise ValueError("`high_res_dem` must be provided when `include_dem_branch` is True.")
        if self.include_dsm_branch and high_res_dsm is None:
            raise ValueError("`high_res_dsm` must be provided when `include_dsm_branch` is True.")

        # Squeeze the time dimension (T=1) from dynamic inputs if present
        if weather.ndim == 5 and weather.shape[1] == 1:
            weather = weather.squeeze(1) # (B, C_weather, H_orig, W_orig)
        elif weather.ndim != 4:
            raise ValueError(f"Weather tensor has unexpected shape {weather.shape}. Expected 4D or 5D with T=1.")

        # Remove time dimension from static_lst as well if it's used and has T=1
        if self.use_lst and static_lst is not None:
            if static_lst.ndim == 5 and static_lst.shape[1] == 1:
                static_lst = static_lst.squeeze(1) # -> (B, C_lst, H_low_res, W_low_res)
            elif static_lst.ndim != 4:
                 raise ValueError(f"static_lst has unexpected shape {static_lst.shape}. Expected 4D or 5D with T=1.")

        # 1. Encode Clay Features Dynamically
        # Respects frozen status set in ClayFeatureExtractor.__init__
        clay_features = self.clay_backbone(cloudless_mosaic, norm_time_tensor, norm_latlon_tensor)
        B, D_clay, H_feat, W_feat = clay_features.shape
        # Project Clay features (always done now before the head)
        projected_clay = self.proj(clay_features) # (B, proj_ch, H', W')

        # 2. Resize Weather and LST (if used) to match Clay feature map size (H', W')
        weather_resized = F.interpolate(weather, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        # Prepare list for concatenation - Content depends on head type
        combined_features_list = []
        # Always use projected clay features now
        combined_features_list.append(projected_clay)
        # Add resized weather
        combined_features_list.append(weather_resized)

        if self.use_lst and static_lst is not None: # Check static_lst exists after potential squeeze
             if static_lst.ndim != 4:
                  raise ValueError(f"static_lst has unexpected number of dimensions ({static_lst.ndim}) before interpolation. Expected 4.")
             
             # Resize LOW-RES LST to match Clay feature map size (H', W')
             static_lst_resized = F.interpolate(static_lst, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
             combined_features_list.append(static_lst_resized) # Append low-res LST

        # 2b. Process High-Res Elevation Branches (if included)
        if self.include_dem_branch and high_res_dem is not None:
            dem_features = self.dem_branch(high_res_dem) # (B, C_elev_out, H_elev_feat, W_elev_feat)
            # Resize to match Clay feature map size
            dem_features_resized = F.interpolate(dem_features, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            combined_features_list.append(dem_features_resized)
            
        if self.include_dsm_branch and high_res_dsm is not None:
            dsm_features = self.dsm_branch(high_res_dsm) # (B, C_elev_out, H_elev_feat, W_elev_feat)
            # Resize to match Clay feature map size
            dsm_features_resized = F.interpolate(dsm_features, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            combined_features_list.append(dsm_features_resized)

        # 3. Concatenate all features
        combined_features = torch.cat(combined_features_list, dim=1)
        # Shape: (B, proj_ch + C_weather [+ C_lst] [+ C_dem] [+ C_dsm], H', W')

        # 4. Pass through Selected Head
        if self.head_type == 'unet':
            # UNet head outputs at its own feature resolution
            prediction_feat_res = self.head(combined_features) # (B, 1, H_unet, W_unet)
        elif self.head_type == 'simple_cnn':
            # SimpleCNN head outputs at feature map resolution
            prediction_feat_res = self.head(combined_features) # (B, 1, H', W')
        else: # Should not happen
            raise ValueError(f"Invalid head_type {self.head_type} during forward pass.")

        # 5. Resize prediction back to the original target size (for both head types)
        if prediction_feat_res.shape[2:] != target_h_w:
            prediction_resized = F.interpolate(prediction_feat_res, size=target_h_w, mode='bilinear', align_corners=False)
        else:
            prediction_resized = prediction_feat_res

        return prediction_resized
