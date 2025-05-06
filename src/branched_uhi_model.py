#!/usr/bin/env python3
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.model import UNetConvBlock, ClayFeatureExtractor
from src.ingest.data_utils import determine_target_grid_size

# -----------------------------------------------------------------------------
# ConvLSTM Implementation -----------------------------------------------------
# -----------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM Cell implementation.

    Based on the architecture described in:
    Shi et al., 2015. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting".
    (https://arxiv.org/abs/1506.04214)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Args:
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden state.
            kernel_size (int or tuple): Size of the convolutional kernel.
            bias (bool): Whether or not to add the bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # Ensure same padding
        self.bias = bias

        # Convolution for input_dim + hidden_dim -> 4 * hidden_dim (for gates i, f, o, g)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        # Initialize weights using orthogonal initialization
        # This helps with gradient stability in recurrent networks
        nn.init.orthogonal_(self.conv.weight)
        if bias:
            # Initialize biases: forget gate bias to 1.0 (helps with learning), others to 0
            nn.init.zeros_(self.conv.bias)
            # Set the biases for the forget gate to 1.0
            # The layout is [input_gate, forget_gate, output_gate, cell_gate]
            # Each is hidden_dim in size
            self.conv.bias.data[self.hidden_dim:2*self.hidden_dim].fill_(1.0)

    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor for time step t (b, c, h, w)
            cur_state (tuple): Containing current hidden and cell state (h_cur, c_cur)
                               both of shape (b, hidden_dim, h, w)
        Returns:
            tuple: next hidden state, next cell state (h_next, c_next)
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (b, input_dim + hidden_dim, h, w)

        combined_conv = self.conv(combined) # (b, 4 * hidden_dim, h, w)

        # Split combined_conv into gates: input, forget, output, cell_input
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute next cell state
        c_next = f * c_cur + i * g
        # Compute next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """Initializes hidden state and cell state with zeros."""
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    """ConvLSTM network implementation.

    Stacks multiple ConvLSTM layers. Handles hidden state initialization and propagation.
    Based on the architecture described in:
    Shi et al., 2015. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting".
    (https://arxiv.org/abs/1506.04214)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False):
        """
        Args:
            input_dim (int): Number of channels in input
            hidden_dim (int or list): Number of hidden channels in each layer.
                                      If int, all layers have same hidden dim.
            kernel_size (int or tuple or list): Kernel size for each layer.
                                                If int/tuple, all layers use same kernel.
            num_layers (int): Number of ConvLSTM layers
            batch_first (bool): If True, then Input/Output shape = (b, t, c, h, w)
            bias (bool): Bias bool.
            return_all_layers (bool): If True, return all layer outputs.
        """
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure kernel_size is a list of tuples
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # Make sure hidden_dim is a list
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor (b, t, c, h, w) or (t, b, c, h, w)
            hidden_state (list of tuple, optional): List of initial hidden states (h, c) per layer.
                                                      Defaults to zeros.
        Returns:
            tuple: layer_output_list, last_state_list
                layer_output_list (list): List of output tensors for each layer, each (b, t, hidden_dim, h, w)
                last_state_list (list): List of last states (h, c) for each layer
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input_tensor shape = (b, t, c, h, w)
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                  cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1) # (b, t, hidden_dim, h, w)
            cur_layer_input = layer_output # Output of current layer is input to next

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.batch_first:
            layer_output_list = [o.permute(1, 0, 2, 3, 4) for o in layer_output_list]

        # Return only the final state tuple (h, c) of the LAST layer
        return last_state_list[-1]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('kernel_size must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """Helper function to expand single param value to a list for all layers."""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class UNetUpBlockInterpolate(nn.Module):
    """UNet Up-block using Interpolate + Conv."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        # Conv block input channels = skip channels (out_channels) + upsampled channels (in_channels)
        self.conv = UNetConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1_upsampled = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # Pad x1_upsampled if needed
        diffY = x2.size()[2] - x1_upsampled.size()[2]
        diffX = x2.size()[3] - x1_upsampled.size()[3]
        x1_upsampled = F.pad(x1_upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate
        x = torch.cat([x2, x1_upsampled], dim=1)
        return self.conv(x)

# -----------------------------------------------------------------------------
# Branched UHI Model ----------------------------------------------------------
# -----------------------------------------------------------------------------

class BranchedUHIModel(nn.Module):
    """
    UHI prediction model with separate branches for temporal (weather) and
    static features, fused before a U-Net style head.
    Uses ConvLSTM for weather processing.
    Accepts ALL spatial features (weather, clay, lst, dem, dsm, indices) at a
    common feature resolution.
    """
    def __init__(self,
                 # --- Weather Branch Config --- #
                 weather_input_channels: int,
                 convlstm_hidden_dims: List[int],
                 convlstm_kernel_sizes: List[Tuple[int, int]],
                 convlstm_num_layers: int,
                 # --- Static Feature Config --- #
                 feature_flags: Dict[str, bool],
                 # --- Head Config --- #
                 proj_static_ch: int, # Projection channels for combined static feats
                 proj_temporal_ch: int, # Projection channels for ConvLSTM output
                 unet_base_channels: int,
                 unet_depth: int,
                 # --- Target Grid Info ---
                 uhi_grid_resolution_m: int,
                 bounds: List[float],
                 # Optional arguments (can have defaults)
                 sentinel_bands_to_load: Optional[List[str]] = None,
                 # Clay Specific (optional, defaults handled internally if use_clay is False)
                 clay_model_size: Optional[str] = None,
                 clay_bands: Optional[List[str]] = None,
                 clay_platform: Optional[str] = None,
                 clay_gsd: Optional[int] = None,
                 freeze_backbone: bool = True,
                 clay_checkpoint_path: Optional[Union[str, Path]] = None,
                 clay_metadata_path: Optional[Union[str, Path]] = None):
        """
        Initializes the Branched UHI Model with common feature resolution.

        Args:
            weather_input_channels (int): Number of input channels for weather data.
            convlstm_hidden_dims (List[int]): List of hidden dimensions for ConvLSTM layers.
            convlstm_kernel_sizes (List[Tuple[int, int]]): List of kernel sizes for ConvLSTM layers.
            convlstm_num_layers (int): Number of ConvLSTM layers.
            feature_flags (Dict[str, bool]): Controls inclusion of static features (use_dem, use_dsm, etc.).
            sentinel_bands_to_load (Optional[List[str]]): Bands if using sentinel_composite.
            clay_model_size (Optional[str]): Size of Clay model if use_clay is True.
            clay_bands (Optional[List[str]]): Bands for Clay if use_clay is True.
            clay_platform (Optional[str]): Platform for Clay if use_clay is True.
            clay_gsd (Optional[int]): GSD for Clay if use_clay is True.
            freeze_backbone (bool): Whether to freeze Clay backbone weights.
            clay_checkpoint_path (Optional[str]): Path to Clay checkpoint if use_clay is True.
            clay_metadata_path (Optional[str]): Path to Clay metadata if use_clay is True.
            proj_static_ch (int): Output channels for static feature projection layer.
            proj_temporal_ch (int): Output channels for temporal feature projection layer.
            unet_base_channels (int): Base number of channels for the U-Net decoder.
            unet_depth (int): Number of down/up sampling blocks in the U-Net decoder.
            uhi_grid_resolution_m (int): The resolution of the final target UHI grid.
            bounds (List[float]): The geographic bounds [min_lon, min_lat, max_lon, max_lat].
        """
        super().__init__()
        self.feature_flags = feature_flags
        self.bounds = bounds
        self.uhi_grid_resolution_m = uhi_grid_resolution_m

        # Calculate and store target output dimensions
        self.target_H, self.target_W = determine_target_grid_size(
            self.bounds, self.uhi_grid_resolution_m
        )
        logging.info(f"Model configured for target output grid size: ({self.target_H}, {self.target_W})")

        # --- Weather Branch (ConvLSTM) --- #
        self.conv_lstm = ConvLSTM(
            input_dim=weather_input_channels,
            hidden_dim=convlstm_hidden_dims,
            kernel_size=convlstm_kernel_sizes,
            num_layers=convlstm_num_layers,
            batch_first=True, # Expect input (B, T, C, H, W)
            bias=True,
            return_all_layers=False
        )

        # --- Static Feature Processing --- #

        # Clay Backbone (Optional)
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

        # Calculate number of channels expected from the dataloader's 'static_features' tensor
        dataloader_static_channels = 0
        if self.feature_flags.get("use_lst", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_dem", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_dsm", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_ndvi", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_ndbi", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_ndwi", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_sentinel_composite", False):
            if not sentinel_bands_to_load: raise ValueError("sentinel_bands_to_load required if use_sentinel_composite=True")
            dataloader_static_channels += len(sentinel_bands_to_load)

        # Calculate total input channels for the static projection layer
        static_input_channels = 0
        if clay_output_channels > 0:
            static_input_channels += clay_output_channels # Add Clay channels if used
        if dataloader_static_channels > 0:
            static_input_channels += dataloader_static_channels # Add channels from the dataloader tensor

        logging.info(f"Calculated dataloader static channels: {dataloader_static_channels}")
        logging.info(f"Total calculated input channels for STATIC projection: {static_input_channels}")
        print(f"[DEBUG MODEL INIT] Initializing static_proj with {static_input_channels} input channels.")

        if static_input_channels > 0:
            self.static_proj = nn.Conv2d(static_input_channels, proj_static_ch, kernel_size=1)
        else:
            self.static_proj = None
            logging.warning("No static features enabled or Clay model not used. Static projection layer will be None.")
            if proj_static_ch > 0:
                 logging.warning(f"proj_static_ch ({proj_static_ch}) > 0 but no static features are input.")
                 proj_static_ch = 0 # Ensure it reflects no static input

        # --- Temporal Feature Projection --- #
        # Project the last hidden state of the ConvLSTM
        temporal_input_channels = convlstm_hidden_dims[-1]
        self.temporal_proj = nn.Conv2d(temporal_input_channels, proj_temporal_ch, kernel_size=1)

        # --- U-Net Decoder Head --- #
        unet_input_channels = proj_static_ch + proj_temporal_ch
        if unet_input_channels == 0:
            raise ValueError("No features projected for U-Net head (static_ch=0, temporal_ch=0). Check feature flags and projections.")

        # Use the new decoder that handles target resizing
        self.unet_decoder = UNetDecoderWithTargetResize(
            in_channels=unet_input_channels,
            base_channels=unet_base_channels,
            depth=unet_depth,
            target_h=self.target_H,
            target_w=self.target_W
        )
        self.final_conv = nn.Conv2d(unet_base_channels, 1, kernel_size=1)

        logging.info(f"BranchedUHIModel initialized. Static Proj In: {static_input_channels}, Out: {proj_static_ch}. Temporal Proj In: {temporal_input_channels}, Out: {proj_temporal_ch}. UNet In: {unet_input_channels}")

    def forward(self, weather_seq: torch.Tensor,
                # --- Optional Static Features (All at feature resolution) --- #
                static_features: Optional[torch.Tensor] = None,
                # --- Optional Clay Inputs (All at feature resolution) --- #
                clay_mosaic: Optional[torch.Tensor] = None,
                norm_latlon: Optional[torch.Tensor] = None,
                norm_timestamp: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Forward pass through the Branched UHI Model.

        Args:
            weather_seq (torch.Tensor): Weather sequence (B, T, C_weather, H_feat, W_feat).
            static_features (Optional[torch.Tensor]): Combined low-res static features
                                                    (LST, DEM, DSM, Indices, Composite)
                                                    (B, C_static_other, H_feat, W_feat).
            clay_mosaic (Optional[torch.Tensor]): Input mosaic for Clay (B, C_clay_in, H_feat, W_feat).
            norm_latlon (Optional[torch.Tensor]): Normalized lat/lon for Clay (B, 2, H_feat, W_feat).
            norm_timestamp (Optional[torch.Tensor]): Normalized timestamp for Clay (B, 1).

        Returns:
            torch.Tensor: Predicted UHI grid (B, 1, H_uhi, W_uhi).
        """

        # --- 1. Temporal Branch (ConvLSTM) --- #
        # Input: (B, T, C, H, W), Output is now just the final state tuple (h, c) of the last layer
        last_state = self.conv_lstm(weather_seq)
        temporal_features = last_state[0] # Get hidden state h from the last layer state tuple
        temporal_projected = self.temporal_proj(temporal_features)
        B, _, H_feat, W_feat = temporal_projected.shape

        # --- 2. Static Branch --- #
        all_static_features_list = []

        # Clay Features (Optional)
        if self.clay_model is not None:
            if clay_mosaic is None or norm_latlon is None or norm_timestamp is None:
                raise ValueError("Clay inputs (mosaic, latlon, timestamp) required when Clay model is enabled.")
            # Clay expects (B, C, H, W), norm_latlon (B, 2, H, W), norm_timestamp (B, 1)
            # Ensure norm_timestamp is broadcastable if needed, Clay handles it internally
            clay_features = self.clay_model(clay_mosaic, norm_latlon, norm_timestamp)
            # clay_features shape: (B, C_clay_out, H_clay, W_clay)

            # --- RESIZE Clay features to match other static features --- #
            if clay_features.shape[-2:] != (H_feat, W_feat):
                logging.debug(f"Resizing Clay features from {clay_features.shape[-2:]} to {(H_feat, W_feat)}")
                clay_features = F.interpolate(
                    clay_features, 
                    size=(H_feat, W_feat), 
                    mode='bilinear', 
                    align_corners=False
                )
            # ----------------------------------------------------------- #
            all_static_features_list.append(clay_features)

        # Other Static Features (Optional)
        if static_features is not None:
             # static_features shape: (B, C_static_other, H_feat, W_feat)
             # Ensure it matches the feature resolution spatial dimensions
             if static_features.shape[-2:] != (H_feat, W_feat):
                 raise ValueError(f"Static features spatial dim {static_features.shape[-2:]} != Temporal branch dim {(H_feat, W_feat)}")
             all_static_features_list.append(static_features)

        # Concatenate all available static features
        if not all_static_features_list:
            # If no static features are provided or enabled, and static_proj exists (input_ch > 0),
            # this indicates a configuration mismatch. The check in init should prevent this.
            # If static_proj is None, we proceed with only temporal features.
            if self.static_proj is not None:
                raise ValueError("Static projection layer exists, but no static features were provided to forward pass.")
            static_projected = torch.zeros(B, 0, H_feat, W_feat, device=temporal_projected.device) # Empty static tensor
        else:
            combined_static = torch.cat(all_static_features_list, dim=1)
            
            if self.static_proj is None:
                # This case should ideally not happen if init checks pass
                raise ValueError("Static features provided, but static projection layer is None. Check config.")
            
            # Check for channel mismatch
            static_proj_in_ch = self.static_proj.weight.shape[1]  # Input channels of Conv2d
            if static_proj_in_ch != combined_static.shape[1]:
                print(f"[ERROR] Channel mismatch! static_proj expects {static_proj_in_ch} channels but combined_static has {combined_static.shape[1]} channels")
                # Print the feature flags for debugging
                print(f"[DEBUG] Feature flags: {self.feature_flags}")
                # Hard requirement since spatial dims already match - we can trace back through stack
                raise ValueError(f"Channel mismatch: static_proj expects {static_proj_in_ch} channels but got {combined_static.shape[1]}")
            
            static_projected = self.static_proj(combined_static)

        # --- 3. Fusion & U-Net Decoder --- #
        # Concatenate projected features
        fused_features = torch.cat([static_projected, temporal_projected], dim=1)

        # U-Net Decoder expects (B, C_fused, H_feat, W_feat)
        unet_output = self.unet_decoder(fused_features)
        # unet_output shape: (B, unet_base_channels, H_feat, W_feat)
        
        # Final 1x1 Convolution
        prediction = self.final_conv(unet_output)
        # prediction shape: (B, 1, H_uhi, W_uhi) - ensured by UNetDecoderWithTargetResize

        return prediction


# --- U-Net Decoder Implementation (Original, potentially used by CNN) ---
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, base_channels, depth):
        super().__init__()
        self.depth = depth
        ch = in_channels
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(UNetConvBlock(ch, base_channels * (2**i)))
            ch = base_channels * (2**i)

        self.middle_conv = UNetConvBlock(ch, ch)

        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            # Input channels = current level channels + skip connection channels
            # Output channels = skip connection channels (channels at the shallower level)
            upsample_in_ch = base_channels * (2**(i+1)) # Channels from deeper layer
            skip_ch = base_channels * (2**i)        # Channels from skip connection
            self.ups.append(UNetUpBlockInterpolate(upsample_in_ch, skip_ch))

        # Final output channels will be base_channels after the last up-block

    def forward(self, x):
        skips = []
        # Down path
        for i, down_block in enumerate(self.downs):
            x = down_block(x)
            if i < self.depth - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
            else:
                # No max pool after the last down block
                skips.append(x)

        # Middle
        x = self.middle_conv(skips.pop()) # Use the deepest feature map

        # Up path
        for up_block in self.ups:
            skip_connection = skips.pop()
            x = up_block(x, skip_connection) # x is from deeper layer, skip_connection is from down path

        return x

# --- NEW U-Net Decoder with Target Size Resizing (for Branched Model) ---
class UNetDecoderWithTargetResize(nn.Module):
    """U-Net Decoder that ensures output matches target H, W using interpolation if needed."""
    def __init__(self, in_channels, base_channels, depth, target_h, target_w):
        super().__init__()
        self.depth = depth
        self.target_h = target_h
        self.target_w = target_w

        ch = in_channels
        self.downs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.downs.append(UNetConvBlock(ch, out_ch))
            if i < depth - 1: # Don't add pool after last down block
                self.down_pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.middle_conv = UNetConvBlock(ch, ch) # Middle block uses final down channels

        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            # Input channels for up-block: channels from deeper layer (x1)
            # Skip channels for up-block: channels from corresponding down layer (x2)
            upsample_in_ch = base_channels * (2**(i+1)) if i < depth - 1 else ch # Handle middle block output
            skip_ch = base_channels * (2**i)
            self.ups.append(UNetUpBlockInterpolate(upsample_in_ch, skip_ch))

        # Final output channels will be base_channels after the last up-block
        logging.info(f"Initialized UNetDecoderWithTargetResize. Target: ({target_h}, {target_w})")

    def forward(self, x):
        skips = []
        # Down path
        for i in range(self.depth):
            x = self.downs[i](x)
            skips.append(x)
            if i < self.depth - 1:
                x = self.down_pools[i](x)

        # Middle (use the last element of skips, which is the output of the last down conv)
        x = self.middle_conv(x)

        # Up path (iterate through skips in reverse, starting from second-to-last)
        for i, up_block in enumerate(self.ups):
            skip_connection = skips[self.depth - 1 - i]
            x = up_block(x, skip_connection)

        # --- Final Resize Check --- #
        _, _, current_h, current_w = x.shape
        if current_h != self.target_h or current_w != self.target_w:
            logging.debug(f"UNetDecoder output ({current_h}, {current_w}) != target ({self.target_h}, {self.target_w}). Interpolating.")
            x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bicubic', align_corners=False)

        return x 