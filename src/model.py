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
    Backbone can optionally be frozen.
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
            freeze_backbone (bool): If True, freezes the Clay backbone weights during training.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        # self.model_size = model_size # Store for reference, but don't pass to init directly
        self.bands = bands
        self.platform = platform
        self.gsd = gsd
        self.freeze_backbone = freeze_backbone

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
        self.model.eval()
        # ----------------------------------------

        # --- Freeze Backbone --- 
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logging.info("Clay backbone frozen.")
            self.model.eval() # Ensure model is in eval mode if frozen
        else:
            logging.info("Clay backbone NOT frozen (trainable).")
            self.model.train() # Set to train mode if not frozen

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
        # --- MODIFIED: Conditional torch.no_grad() ---
        context_manager = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context_manager:
            # Set model mode based on freeze flag (eval if frozen, train if not)
            # Note: model.eval() was already called in __init__, but good practice
            # to ensure correct mode, especially if fine-tuning.
            self.model.train(not self.freeze_backbone) 
            
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
    hidden states. 
    MODIFIED: Encodes Clay features dynamically at each step using time/latlon metadata.
    Concatenates dynamic Clay, weather, and time embedding for GRU input.
    """
    def __init__(self,
                 # Clay args
                 clay_checkpoint_path: str,
                 clay_metadata_path: str,
                 # Weather args
                 weather_channels: int, # Num channels in weather_seq input
                 # Time embedding args (minute-based for GRU)
                 time_embed_dim: int = 2,
                 # --- Args with defaults ---
                 proj_ch: int = 32, # Channels after projecting Clay features
                 clay_model_size: str = "large",
                 clay_bands: list = ["blue", "green", "red", "nir"], # Bands for Clay input mosaic
                 clay_platform: str = "sentinel-2-l2a",
                 clay_gsd: int = 10,
                 # LST args - LST is treated as static, projected once
                 lst_channels: int = 1, # Num channels in static LST map
                 use_lst: bool = True,
                 # ConvGRU args
                 gru_hidden_dim: int = 64, # Hidden dimension for ConvGRU cell
                 gru_kernel_size: int = 3,
                 freeze_backbone: bool = True,
    ):
        super().__init__()
        self.use_lst = use_lst
        self.proj_ch = proj_ch
        self.gru_hidden_dim = gru_hidden_dim
        self.weather_channels = weather_channels
        self.time_embed_dim = time_embed_dim # Minute-based embedding
        self.lst_channels = lst_channels

        # --- Clay Feature Extraction (Initialized but used dynamically) ---
        self.clay_backbone = ClayFeatureExtractor(
            checkpoint_path=clay_checkpoint_path,
            metadata_path=clay_metadata_path,
            model_size=clay_model_size,
            bands=clay_bands,
            platform=clay_platform,
            gsd=clay_gsd,
            freeze_backbone=freeze_backbone
        )

        clay_embed_dim = self.clay_backbone.embed_dim
        self.proj = nn.Conv2d(clay_embed_dim, self.proj_ch, kernel_size=1)

        # --- LST Projection (if used) ---
        # Project LST separately if used, as it's static within a sequence
        self.proj_lst = None
        if self.use_lst:
            # Project LST to match proj_ch for easier concatenation later if needed,
            # or handle channel addition differently. Let's project it.
            # We might need a different projection size if we concatenate it differently.
            # For simplicity, let's assume LST is handled before GRU or added separately.
            # MODIFYING: LST will be handled OUTSIDE the step, pre-projected.
            self.proj_lst = nn.Conv2d(self.lst_channels, self.proj_ch, kernel_size=1) # Example projection

        # --- MODIFIED: Calculate GRU Input Channels ---
        # Dynamic Clay (projected) + Weather + Time Embedding (minute-based)
        # If LST is used, it's added separately in the step method after projection
        gru_in_ch = self.proj_ch + self.weather_channels + self.time_embed_dim
        if self.use_lst:
             # We will concatenate the projected LST within the step method
             gru_in_ch += self.proj_ch # Add channels for projected LST


        # --- Recurrent Core ---
        self.gru = ConvGRUCell(in_ch=gru_in_ch, hid_ch=self.gru_hidden_dim, kernel_size=gru_kernel_size)

        # --- Prediction Head ---
        self.regressor = nn.Conv2d(self.gru_hidden_dim, 1, kernel_size=1)

        logging.info(f"UHINet initialized (Dynamic Clay features per step):")
        logging.info(f"  Clay Embed Dim: {clay_embed_dim} -> Proj Dim: {self.proj_ch}")
        logging.info(f"  Use LST: {self.use_lst} (Channels: {self.lst_channels if self.use_lst else 0}) - LST processed statically.")
        # Removed Static/Dynamic log lines as Clay is now dynamic
        logging.info(f"  GRU Input Dim (Proj Dynamic Clay [+ Proj LST] + Weather + TimeEmb): {gru_in_ch}")
        logging.info(f"  GRU Hidden Dim: {self.gru_hidden_dim}")


    # --- REMOVED encode_and_project_static method ---
    # def encode_and_project_static(self, ...)

    # +++ ADDED encode_and_project_lst_static +++
    def encode_and_project_lst_static(self, static_lst: Optional[torch.Tensor], target_feat_h_w: Tuple[int, int]) -> Optional[torch.Tensor]:
        """
        Projects static LST map to match feature map size.
        Should be called once before the time loop if LST is used.

        Args:
            static_lst (torch.Tensor, optional): Static LST map (B, 1, C_lst, H, W).
                                                Provide only if self.use_lst is True.
            target_feat_h_w (Tuple[int, int]): Target spatial dimensions (H', W') from Clay features.

        Returns:
            torch.Tensor or None: Projected static LST features (B, proj_ch, H', W') or None.
        """
        if not self.use_lst or static_lst is None or self.proj_lst is None:
            return None

        # Squeeze time dimension (T=1)
        static_lst = static_lst.squeeze(1) # (B, C_lst, H, W)

        # Resize LST to match Clay feature map size (H', W')
        if static_lst.shape[2:] != target_feat_h_w:
             static_lst_resized = F.interpolate(static_lst, size=target_feat_h_w, mode='bilinear', align_corners=False)
        else:
             static_lst_resized = static_lst

        # Project LST features -> (B, proj_ch, H', W')
        projected_lst = self.proj_lst(static_lst_resized)
        return projected_lst

    # --- REVISED step method ---
    def step(self, 
             sentinel_mosaic: torch.Tensor, # Static mosaic input (B, C, H, W)
             norm_time_tensor: torch.Tensor, # Dynamic normalized time for Clay (B, 4)
             norm_latlon_tensor: torch.Tensor, # Dynamic normalized lat/lon for Clay (B, 4)
             weather_t: torch.Tensor,       # Weather for current step t (B, C_weather, H, W)
             time_emb_t: torch.Tensor,      # Minute-based time emb for step t (B, C_time, H, W)
             h_prev: torch.Tensor,           # Previous hidden state (B, gru_hidden_dim, H', W')
             projected_lst_static: Optional[torch.Tensor] = None # Pre-projected static LST (B, proj_ch, H', W')
            ) -> torch.Tensor:
        """
        Performs a single ConvGRU step with dynamic Clay feature extraction.

        Args:
            sentinel_mosaic: The static cloudless mosaic.
            norm_time_tensor: Normalized time metadata for the current step t.
            norm_latlon_tensor: Normalized lat/lon metadata (usually fixed for sequence).
            weather_t: Weather data for the current step t.
            time_emb_t: Minute-based time embedding for the current step t.
            h_prev: Hidden state from the previous time step t-1.
            projected_lst_static: Optional pre-projected static LST features.

        Returns:
            torch.Tensor: Updated hidden state h_t (B, gru_hidden_dim, H', W').
        """
        
        # 1. Extract Dynamic Clay features using current step's metadata
        # Warning: This is computationally expensive!
        with torch.no_grad(): # Keep backbone frozen
            clay_features_t = self.clay_backbone(sentinel_mosaic, norm_time_tensor, norm_latlon_tensor)
        B, D_clay, H_feat, W_feat = clay_features_t.shape

        # 2. Project Dynamic Clay features
        projected_clay_t = self.proj(clay_features_t) # (B, proj_ch, H', W')
        
        # 3. Check hidden state dimensions
        if h_prev.shape[2:] != (H_feat, W_feat):
             raise ValueError(f"Spatial dimensions of h_prev ({h_prev.shape[2:]}) must match Clay features ({H_feat, W_feat})")

        # 4. Resize weather and time_emb to match Clay feature map size (H', W')
        if weather_t.shape[2:] != (H_feat, W_feat):
            weather_t_resized = F.interpolate(weather_t, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        else:
            weather_t_resized = weather_t
        
        if time_emb_t.shape[2:] != (H_feat, W_feat):
            time_emb_t_resized = F.interpolate(time_emb_t, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        else:
            time_emb_t_resized = time_emb_t
            
        # 5. Concatenate features for GRU input
        # Input: Projected Dynamic Clay, Resized Weather, Resized TimeEmb
        gru_input_list = [projected_clay_t, weather_t_resized, time_emb_t_resized]
        
        # --- MODIFIED: Add static projected LST if available --- 
        if self.use_lst and projected_lst_static is not None:
           # Ensure LST features have the same spatial dimensions
           if projected_lst_static.shape[2:] == projected_clay_t.shape[2:]:
                # Concatenate along the channel dimension (dim=1)
                # Insert after projected Clay, before weather/time for logical grouping
                gru_input_list.insert(1, projected_lst_static) 
           else:
               # This should ideally not happen if target_feat_h_w was used correctly
               logging.warning(f"Projected LST static features dimensions mismatch ({projected_lst_static.shape[2:]} vs {projected_clay_t.shape[2:]}), skipping concatenation.")
        # --- END MODIFIED --- 
                
        x_t_combined = torch.cat(gru_input_list, dim=1)

        # 6. Perform GRU step
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

# -----------------------------------------------------------------------------
# 4. SIMPLE CNN UHI NET MODEL (FOR DEBUGGING) -------------------------------
# -----------------------------------------------------------------------------

class UHINetCNN(nn.Module):
    """
    Simpler feedforward CNN UHI prediction model for debugging.
    MODIFIED: Encodes Clay features dynamically using time/latlon metadata.
    Combines dynamic Clay, static LST (optional), and dynamic weather features,
    then passes them through a CNN.
    """
    def __init__(self,
                 # Clay args (same as UHINet)
                 clay_checkpoint_path: str,
                 clay_metadata_path: str,
                 weather_channels: int,
                 freeze_backbone: bool = True,
                 # --- Args with defaults ---
                 proj_ch: int = 32, # Channels after projecting Clay features
                 clay_model_size: str = "large",
                 clay_bands: list = ["blue", "green", "red", "nir"],
                 clay_platform: str = "sentinel-2-l2a",
                 clay_gsd: int = 10,
                 # LST args
                 lst_channels: int = 1,
                 use_lst: bool = True,
                 # CNN specific args
                 cnn_hidden_dims: List[int] = [64, 32], # Hidden dimensions for CNN layers
                 cnn_kernel_size: int = 3,
                 cnn_dropout: float = 0.0 # Add dropout parameter
    ):
        super().__init__()
        self.use_lst = use_lst
        self.proj_ch = proj_ch
        self.weather_channels = weather_channels
        self.lst_channels = lst_channels
        self.cnn_dropout = cnn_dropout # Store dropout rate

        # --- Dynamic Clay Feature Extraction (Same as UHINet) ---
        self.clay_backbone = ClayFeatureExtractor(
            checkpoint_path=clay_checkpoint_path,
            metadata_path=clay_metadata_path,
            model_size=clay_model_size,
            bands=clay_bands,
            platform=clay_platform,
            gsd=clay_gsd,
            freeze_backbone=freeze_backbone
        )

        clay_embed_dim = self.clay_backbone.embed_dim
        self.proj = nn.Conv2d(clay_embed_dim, self.proj_ch, kernel_size=1)

        # --- Calculate CNN Input Channels ---
        cnn_in_ch = self.proj_ch + self.weather_channels
        if self.use_lst:
            cnn_in_ch += self.lst_channels
        
        # --- CNN Core (Dynamically creates layers with BatchNorm) --- 
        cnn_layers = []
        current_channels = cnn_in_ch
        padding = cnn_kernel_size // 2
        for i, hidden_dim in enumerate(cnn_hidden_dims):
            cnn_layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=cnn_kernel_size, padding=padding))
            # Add BatchNorm after Conv2d
            cnn_layers.append(nn.BatchNorm2d(hidden_dim))
            cnn_layers.append(nn.ReLU(inplace=True))
            # Dropout is applied *after* the core, before the regressor
            current_channels = hidden_dim
        self.cnn_core = nn.Sequential(*cnn_layers)

        # --- Add BatchNorm before the final regressor --- 
        self.final_bn = nn.BatchNorm2d(current_channels)
        # -------------------------------------------------

        # --- Prediction Head --- 
        self.regressor = nn.Conv2d(current_channels, 1, kernel_size=1)

        logging.info(f"UHINetCNN initialized (Feedforward CNN with Dynamic Clay):")
        logging.info(f"  Clay Embed Dim: {clay_embed_dim} -> Proj Dim: {self.proj_ch}")
        logging.info(f"  Use LST: {self.use_lst} (Channels: {self.lst_channels if self.use_lst else 0})")
        logging.info(f"  CNN Input Dim (Proj Dyn Clay [+ LST] + Weather): {cnn_in_ch}")
        logging.info(f"  CNN Hidden Dims: {cnn_hidden_dims} (with BatchNorm)")
        logging.info(f"  CNN Dropout Rate: {self.cnn_dropout}") # Log dropout
        logging.info(f"  CNN Output Dim (Before Final BN/Regressor): {current_channels}")

    # --- MODIFIED forward method ---
    def forward(self, cloudless_mosaic: torch.Tensor,
                norm_time_tensor: torch.Tensor,  # Added
                norm_latlon_tensor: torch.Tensor, # Added
                weather: torch.Tensor,
                target_h_w: Tuple[int, int], 
                static_lst: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the CNN model using dynamic Clay features.

        Args:
            cloudless_mosaic (torch.Tensor): Cloudless mosaic (B, C_clay, H_orig, W_orig).
            norm_time_tensor (torch.Tensor): Normalized time tensor for Clay (B, 4).
            norm_latlon_tensor (torch.Tensor): Normalized lat/lon tensor for Clay (B, 4).
            weather (torch.Tensor): Weather data for the time step (B, 1, C_weather, H_orig, W_orig).
            target_h_w (Tuple[int, int]): The desired output height and width (H_orig, W_orig).
            static_lst (torch.Tensor, optional): Static LST map (B, 1, C_lst, H_orig, W_orig).
                                                Provide only if self.use_lst is True.

        Returns:
            torch.Tensor: Predicted UHI map (B, 1, H_orig, W_orig).
        """
        # 0. Input Checks and Preparation
        if self.use_lst and static_lst is None:
            raise ValueError("`static_lst` must be provided when `use_lst` is True.")
        if not self.use_lst and static_lst is not None:
            logging.warning("`static_lst` provided but `use_lst` is False. LST will be ignored.")

        # Squeeze the time dimension (T=1) from dynamic inputs
        weather = weather.squeeze(1) # (B, C_weather, H_orig, W_orig)
        # Remove time dimension from static_lst as well if it's used
        if self.use_lst and static_lst is not None:
            if static_lst.ndim == 5 and static_lst.shape[1] == 1:
                static_lst = static_lst.squeeze(1) # -> (B, C_lst, H_orig, W_orig)
            elif static_lst.ndim != 4:
                # If it's already 4D or some other unexpected shape, raise error
                 raise ValueError(f"static_lst has unexpected shape {static_lst.shape} after potentially squeezing time dim. Expected 4D (B, C, H, W).")

        # 1. Encode Clay Features Dynamically
        with torch.no_grad(): # Assume backbone frozen for CNN version
            clay_features = self.clay_backbone(cloudless_mosaic, norm_time_tensor, norm_latlon_tensor)
        B, _, H_feat, W_feat = clay_features.shape
        projected_clay = self.proj(clay_features) # (B, proj_ch, H', W')

        # 2. Resize Weather and LST (if used) to match Clay feature map size (H', W')
        weather_resized = F.interpolate(weather, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        # Prepare list for concatenation
        combined_features_list = [projected_clay, weather_resized]

        if self.use_lst:
             # static_lst should now be 4D (B, C_lst, H_orig, W_orig)
             if static_lst.ndim != 4:
                  # This check should ideally not be needed if squeeze above worked
                  raise ValueError(f"static_lst has unexpected number of dimensions ({static_lst.ndim}) before interpolation. Expected 4.")
             
             # Resize LST to match Clay feature map size (H', W')
             static_lst_resized = F.interpolate(static_lst, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
             # Insert LST after projected Clay
             combined_features_list.insert(1, static_lst_resized)

        # 3. Concatenate all features
        combined_features = torch.cat(combined_features_list, dim=1)
        # Shape: (B, proj_ch [+ C_lst] + C_weather, H', W')

        # 4. Pass through CNN Core
        cnn_output = self.cnn_core(combined_features)

        # --- Apply Final BatchNorm --- 
        cnn_output = self.final_bn(cnn_output)
        # -----------------------------

        # --- Apply Dropout (after BN, before Regressor) --- 
        if self.cnn_dropout > 0.0 and self.training:
             cnn_output = F.dropout(cnn_output, p=self.cnn_dropout, training=self.training)
        # ---------------------------------------------------

        # 5. Prediction Head
        prediction = self.regressor(cnn_output) # (B, 1, H', W')

        # 6. Resize prediction back to the original target size
        prediction_resized = F.interpolate(prediction, size=target_h_w, mode='bilinear', align_corners=False)

        return prediction_resized
