import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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

        # --- ADDED: Assign output channels ---
        self.output_channels = self.embed_dim
        logging.info(f"ClayFeatureExtractor output channels set to: {self.output_channels}")
        # --- END ADDED ---

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
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Use better weight initialization (Kaiming/He initialization)
        # This helps with gradient flow in deep networks
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
            
        # Initialize batch norm to be identity function initially
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # Only apply dropout during training
        if self.training:
            x = self.dropout(x)
        return x

class UNetUpBlock(nn.Module):
    """Helper: Upsample(ConvTranspose2d 2x2, stride=2) -> Concat -> UNetConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels: Channels from the deeper layer (input to ConvTranspose2d)
        # out_channels: Channels from the skip connection (and the desired output channels for this block)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # The input to the subsequent conv block is the concatenation of:
        # 1. Skip connection channels (out_channels)
        # 2. Upsampled channels (in_channels // 2)
        conv_in_channels = out_channels + (in_channels // 2)
        self.conv = UNetConvBlock(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: input from previous (deeper) layer (to be upsampled)
        x2: input from corresponding skip connection (encoder)
        """
        x1 = self.up(x1) # Shape becomes (B, C1=in_channels//2, H1, W1)
        # x2 shape is (B, C2=out_channels, H2, W2)
        
        # Target spatial dimensions are those of x1 (the upsampled tensor)
        target_h, target_w = x1.shape[2], x1.shape[3]
        h2, w2 = x2.shape[2], x2.shape[3]

        # Handle potential size mismatch. We want the final concatenated tensor
        # to have the spatial dimensions of x1 (H1, W1).
        if target_h != h2 or target_w != w2:
            logging.debug(f"Size mismatch in UNetUpBlock: x1({x1.shape}), x2({x2.shape}). Adjusting x2.")
            # Calculate the difference needed to make x2 match x1
            diff_y = target_h - h2
            diff_x = target_w - w2

            if diff_y >= 0 and diff_x >= 0:
                # Case 1: x1 is larger or equal. Pad x2.
                pad_top = diff_y // 2
                pad_bottom = diff_y - pad_top
                pad_left = diff_x // 2
                pad_right = diff_x - pad_left
                x2 = F.pad(x2, [pad_left, pad_right, pad_top, pad_bottom])
                logging.debug(f"Padded x2 to size: {x2.shape}")
            elif diff_y <= 0 and diff_x <= 0:
                # Case 2: x2 is larger. Crop x2.
                # Note: Using absolute values of potentially negative differences
                crop_top = abs(diff_y) // 2
                crop_bottom = abs(diff_y) - crop_top
                crop_left = abs(diff_x) // 2
                crop_right = abs(diff_x) - crop_left
                x2 = x2[:, :, crop_top : h2 - crop_bottom, crop_left : w2 - crop_right]
                logging.debug(f"Cropped x2 to size: {x2.shape}")
            else:
                # Mixed case (one dim larger, one smaller) - this is problematic
                # Indicates a potential issue earlier in the network.
                # Fallback: Resize x2 to match x1 using interpolation.
                logging.warning(f"Mixed size mismatch in UNetUpBlock (x1={x1.shape}, x2={x2.shape}). Resizing x2.")
                x2 = F.interpolate(x2, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # Now x1 and x2 *should* have matching spatial dimensions (target_h, target_w)
        # Add a final assertion for safety during development/debugging
        assert x1.shape[2:] == x2.shape[2:], \
               f"UNetUpBlock FAILED to align shapes! x1: {x1.shape}, x2: {x2.shape}"

        # Concatenate along channel dimension: (B, C2 + C1, H1, W1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- U-Net Decoder Base Implementation ---
class UNetDecoder(nn.Module):
    """Standard U-Net decoder implementation."""
    def __init__(self, in_channels, base_channels, depth):
        super().__init__()
        self.depth = depth
        
        # Down path
        ch = in_channels
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.downs.append(UNetConvBlock(ch, out_ch))
            if i < depth - 1:  # Don't pool after last down block
                self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = UNetConvBlock(ch, ch)

        # Up path
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch = ch
            out_ch = base_channels * (2**i)
            self.ups.append(UNetUpBlock(in_ch, out_ch))
            ch = out_ch

        logging.info(f"Initialized UNetDecoder. In channels: {in_channels}, Base channels: {base_channels}, Depth: {depth}")

    def forward(self, x):
        # Store skip connections
        skips = []
        
        # Down path
        for i in range(self.depth):
            x = self.downs[i](x)
            skips.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Up path
        for i, up_block in enumerate(self.ups):
            skip_idx = self.depth - 1 - i  # Index into skips list
            skip = skips[skip_idx]
            x = up_block(x, skip)
            
        return x

# --- U-Net Decoder with Target Size Resizing ---
class UNetDecoderWithTargetResize(UNetDecoder):
    """U-Net Decoder that ensures output matches target H, W using a learnable upsampling stage."""
    def __init__(self, in_channels, base_channels, depth, target_h, target_w):
        super().__init__(in_channels, base_channels, depth)
        self.target_h = target_h
        self.target_w = target_w
        
        # Lightweight learnable upsampling stage to reach the final target size
        # Takes the output of the main U-Net decoder (base_channels)
        self.final_upsampler = nn.Sequential(
            nn.Upsample(size=(target_h, target_w), mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels), # Added BatchNorm
            nn.ReLU(inplace=True)
        )
        
        logging.info(f"Initialized UNetDecoderWithTargetResize. Base U-Net output channels: {base_channels}. Target: ({target_h}, {target_w})")

    def forward(self, x):
        # Log input shape
        logging.debug(f"UNetDecoder input shape: {x.shape}")
        
        # Store skip connections
        skips = []
        
        # Down path (inherited from UNetDecoder)
        for i in range(self.depth):
            x = self.downs[i](x)
            skips.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
        
        # Bottleneck (inherited from UNetDecoder)
        x = self.bottleneck(x)
        logging.debug(f"UNetDecoder bottleneck shape: {x.shape}")

        # Up path (inherited from UNetDecoder)
        for i, up_block in enumerate(self.ups):
            skip_idx = self.depth - 1 - i  # Index into skips list
            skip = skips[skip_idx]
            logging.debug(f"UNetDecoder up level {i}: x shape={x.shape}, skip shape={skip.shape}")
            x = up_block(x, skip)

        # Apply the final learnable upsampling stage
        x = self.final_upsampler(x)
        logging.debug(f"UNetDecoder final output shape after upsampler: {x.shape}")
        
        # Final check (should match target size now)
        _, _, h, w = x.shape
        if h != self.target_h or w != self.target_w:
             # This should ideally not happen with the explicit nn.Upsample
             logging.warning(f"Output size {h}x{w} STILL mismatch target {self.target_h}x{self.target_w} after final upsampler! Check architecture.")
             # Fallback resize just in case
             x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)

        return x

# -----------------------------------------------------------------------------
# 5. CNN-BASED UHI NET MODEL  ----------------------------
# -----------------------------------------------------------------------------

class UHINetCNN(nn.Module):
    """
    UHI Prediction CNN model with optional Clay integration.
    Accepts ALL spatial features (weather, clay, lst, dem, dsm, indices) at a
    common feature resolution.
    """
    def __init__(self,
                 # --- Input Feature Config --- #
                 feature_flags: Dict[str, bool], # To determine input channels
                 weather_channels: int, # Number of weather channels
                 sentinel_bands_to_load: Optional[List[str]] = None, # For composite
                 # Clay Specific (if feature_flags["use_clay"])
                 clay_model_size: Optional[str] = None,
                 clay_bands: Optional[List[str]] = None,
                 clay_platform: Optional[str] = None,
                 clay_gsd: Optional[int] = None,
                 freeze_backbone: bool = True,
                 clay_checkpoint_path: Optional[str] = None,
                 clay_metadata_path: Optional[str] = None,
                 # --- CNN Backbone Config --- #
                 base_channels: int = 64,
                 depth: int = 4,
                 # --- REMOVED Elevation Branch Config --- #
                ):
        """
        Initializes the UHINetCNN Model with common feature resolution.

        Args:
            feature_flags (Dict[str, bool]): Controls inclusion of input features.
            weather_channels (int): Number of channels in the weather grid input.
            sentinel_bands_to_load (Optional[List[str]]): Bands if using sentinel_composite.
            clay_model_size (Optional[str]): Size of Clay model if use_clay is True.
            clay_bands (Optional[List[str]]): Bands for Clay if use_clay is True.
            clay_platform (Optional[str]): Platform for Clay if use_clay is True.
            clay_gsd (Optional[int]): GSD for Clay if use_clay is True.
            freeze_backbone (bool): Whether to freeze Clay backbone weights.
            clay_checkpoint_path (Optional[str]): Path to Clay checkpoint if use_clay is True.
            clay_metadata_path (Optional[str]): Path to Clay metadata if use_clay is True.
            base_channels (int): Base number of channels for the U-Net.
            depth (int): Depth of the U-Net.
        """
        super().__init__()
        self.feature_flags = feature_flags
        self.depth = depth

        # --- Clay Backbone (Optional) --- #
        self.clay_model = None
        clay_output_channels = 0
        if self.feature_flags.get("use_clay", False):
            if not all([clay_checkpoint_path, clay_metadata_path, clay_model_size, clay_bands, clay_platform, clay_gsd]):
                 raise ValueError("Missing required Clay configuration parameters when use_clay=True.")
            self.clay_model = ClayFeatureExtractor(
            model_size=clay_model_size,
            bands=clay_bands,
            platform=clay_platform,
            gsd=clay_gsd,
                freeze_backbone=freeze_backbone,
                checkpoint_path=clay_checkpoint_path,
                metadata_path=clay_metadata_path,
            )
            clay_output_channels = self.clay_model.output_channels
            logging.info(f"Initialized Clay model ({clay_model_size}), output channels: {clay_output_channels}")

        # --- Calculate Input Channels for U-Net --- #
        # Start with weather channels
        input_channels = weather_channels
        # Add Clay output channels (if used)
        input_channels += clay_output_channels
        # Add channels for other enabled static features
        if self.feature_flags.get("use_lst", False): input_channels += 1
        if self.feature_flags.get("use_dem", False): input_channels += 1
        if self.feature_flags.get("use_dsm", False): input_channels += 1
        if self.feature_flags.get("use_ndvi", False): input_channels += 1
        if self.feature_flags.get("use_ndbi", False): input_channels += 1
        if self.feature_flags.get("use_ndwi", False): input_channels += 1
        if self.feature_flags.get("use_sentinel_composite", False) and sentinel_bands_to_load:
            input_channels += len(sentinel_bands_to_load)

        if input_channels == 0:
             raise ValueError("No input features selected for UHINetCNN (weather, clay, static features all disabled/zero channels).")
        logging.info(f"Total calculated input channels for U-Net: {input_channels}")

        # --- U-Net Architecture (using centralized implementation) --- #
        self.unet_decoder = UNetDecoder(
            in_channels=input_channels,
            base_channels=base_channels,
            depth=depth
        )
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

        logging.info(f"UHINetCNN initialized. U-Net Input Ch: {input_channels}, Base Ch: {base_channels}, Depth: {depth}")

    def forward(self, weather: torch.Tensor,
                # --- Optional Static Features (All at feature resolution) --- #
                static_features: Optional[torch.Tensor] = None,
                # --- Optional Clay Inputs (All at feature resolution) --- #
                clay_mosaic: Optional[torch.Tensor] = None,
                norm_latlon: Optional[torch.Tensor] = None,
                norm_timestamp: Optional[torch.Tensor] = None,
               ) -> torch.Tensor:
        """
        Forward pass through the UHINetCNN Model.

        Args:
            weather (torch.Tensor): Weather grid (B, C_weather, H_feat, W_feat).
            static_features (Optional[torch.Tensor]): Combined low-res static features
                                                    (LST, DEM, DSM, Indices, Composite)
                                                    (B, C_static_other, H_feat, W_feat).
            clay_mosaic (Optional[torch.Tensor]): Input mosaic for Clay (B, C_clay_in, H_feat, W_feat).
            norm_latlon (Optional[torch.Tensor]): Normalized lat/lon for Clay (B, 2, H_feat, W_feat).
            norm_timestamp (Optional[torch.Tensor]): Normalized timestamp for Clay (B, 1).

        Returns:
            torch.Tensor: Predicted UHI grid (B, 1, H_uhi, W_uhi).
        """
        B, _, H_feat, W_feat = weather.shape
        all_features_list = [weather]

        # --- Clay Features (Optional) --- #
        if self.clay_model is not None:
            if clay_mosaic is None or norm_latlon is None or norm_timestamp is None:
                raise ValueError("Clay inputs (mosaic, latlon, timestamp) required when Clay model is enabled.")
            if clay_mosaic.shape[-2:] != (H_feat, W_feat):
                 raise ValueError(f"Clay mosaic spatial dim {clay_mosaic.shape[-2:]} != Weather dim {(H_feat, W_feat)}")
            
            # clay_features will be (B, D_clay, H_patch, W_patch), e.g. (B, 1024, 14, 14)
            clay_features_raw = self.clay_model(clay_mosaic, norm_latlon, norm_timestamp)

            # Explicitly interpolate clay_features_raw to match H_feat, W_feat
            if clay_features_raw.shape[-2:] != (H_feat, W_feat):
                clay_features_interpolated = F.interpolate(
                    clay_features_raw,
                    size=(H_feat, W_feat),
                    mode='bilinear',
                    align_corners=False
                )
                logging.debug(f"Interpolated Clay output from {clay_features_raw.shape} to {clay_features_interpolated.shape}")
                all_features_list.append(clay_features_interpolated)
            else:
                all_features_list.append(clay_features_raw) # Already matching

        # --- Other Static Features (Optional) --- #
        if static_features is not None:
             if static_features.shape[-2:] != (H_feat, W_feat):
                 raise ValueError(f"Static features spatial dim {static_features.shape[-2:]} != Weather dim {(H_feat, W_feat)}")
             all_features_list.append(static_features)

        # --- Combine all features --- #
        x = torch.cat(all_features_list, dim=1)

        # --- Use U-Net Decoder --- #
        x = self.unet_decoder(x)
        
        # Final convolution to get output
        prediction = self.final_conv(x)

        return prediction
