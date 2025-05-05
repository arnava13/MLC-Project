#!/usr/bin/env python3
import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.model import UNetConvBlock, ClayFeatureExtractor, HighResElevationBranch # Re-use UNetConvBlock, ClayExtractor AND HighResElevationBranch

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

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
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

        # We only need the output sequence of the last layer and the final state
        # Return format matches LSTM: (output_seq_last_layer, (last_h, last_c))
        # Note: last_h and last_c will be lists containing the states for each layer
        last_h_states = [state[0] for state in last_state_list]
        last_c_states = [state[1] for state in last_state_list]

        # Return last layer's output sequence and tuple of final hidden/cell states (stacked over layers)
        return layer_output_list[-1], (torch.stack(last_h_states, dim=0), torch.stack(last_c_states, dim=0))

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
    Optionally includes high-resolution elevation branches processed separately.
    """
    def __init__(self,
                 weather_input_channels: int,
                 convlstm_hidden_dims: List[int],
                 convlstm_kernel_sizes: List[Tuple[int, int]],
                 convlstm_num_layers: int,
                 static_channels: int, # Number of NON-ELEVATION, NON-CLAY static channels
                 unet_base_channels: int,
                 unet_depth: int,
                 convlstm_batch_first: bool = True,
                 include_clay_features: bool = False,
                 clay_checkpoint_path: Optional[str] = None,
                 clay_metadata_path: Optional[str] = None,
                 freeze_clay_backbone: bool = True,
                 clay_embed_dim: Optional[int] = 1024,
                 proj_static_ch: int = 32, # For projecting non-clay, non-elevation static features
                 proj_temporal_ch: int = 32,
                 # --- NEW: High-Resolution Elevation Branch Args --- #
                 include_dem_branch: bool = False,
                 include_dsm_branch: bool = False,
                 elevation_branch_start_channels: int = 16,
                 elevation_branch_out_channels: int = 32,
                 elevation_branch_downsample_layers: int = 4,
                 elevation_branch_kernel_size: int = 3,
                 # ---------------------------------------------------- #
                 **clay_kwargs
                 ):
        super().__init__()
        self.weather_input_channels = weather_input_channels
        self.static_channels = static_channels # Non-elev, non-clay
        self.include_clay_features = include_clay_features and clay_checkpoint_path and clay_metadata_path
        self.clay_embed_dim = clay_embed_dim
        # --- Store elevation flags --- #
        self.include_dem_branch = include_dem_branch
        self.include_dsm_branch = include_dsm_branch
        self.elevation_out_channels = elevation_branch_out_channels
        # ---------------------------- #

        # --- Weather Branch (ConvLSTM) ---
        # Small spatial feature extractor applied before ConvLSTM
        self.weather_feature_extractor = nn.Sequential(
            nn.Conv2d(weather_input_channels, weather_input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(weather_input_channels),
            nn.ReLU(inplace=True)
        )
        convlstm_input_dim = weather_input_channels

        self.convlstm = ConvLSTM(
            input_dim=convlstm_input_dim,
            hidden_dim=convlstm_hidden_dims,
            kernel_size=convlstm_kernel_sizes,
            num_layers=convlstm_num_layers,
            batch_first=convlstm_batch_first,
            bias=True
        )
        # Output channels from the weather branch is the hidden dim of the last ConvLSTM layer
        temporal_output_channels = convlstm_hidden_dims[-1]

        # --- Static Branch ---
        self.clay_backbone = None
        if self.include_clay_features:
            if self.clay_embed_dim is None:
                 raise ValueError("clay_embed_dim must be provided if include_clay_features is True")
            self.clay_backbone = ClayFeatureExtractor(
                checkpoint_path=clay_checkpoint_path,
                metadata_path=clay_metadata_path,
                freeze_backbone=freeze_clay_backbone,
                **clay_kwargs
            )
            # Ensure Clay embed dim matches if provided
            if self.clay_embed_dim != self.clay_backbone.embed_dim:
                logging.warning(f"Provided clay_embed_dim ({self.clay_embed_dim}) does not match inferred dim ({self.clay_backbone.embed_dim}) from Clay model. Using inferred dim.")
                self.clay_embed_dim = self.clay_backbone.embed_dim

        # Projection layer for NON-ELEVATION static features (Original Static + Optional Clay)
        non_elevation_static_input_ch = static_channels + (self.clay_embed_dim if self.include_clay_features else 0)
        self.proj_non_elev_static = nn.Conv2d(non_elevation_static_input_ch, proj_static_ch, kernel_size=1) if non_elevation_static_input_ch > 0 else None

        # --- NEW: High-Resolution Elevation Branches ---
        self.dem_branch = None
        if self.include_dem_branch:
            self.dem_branch = HighResElevationBranch(
                in_channels=1,
                start_channels=elevation_branch_start_channels,
                out_channels=self.elevation_out_channels,
                num_downsample_layers=elevation_branch_downsample_layers,
                kernel_size=elevation_branch_kernel_size
            )
        self.dsm_branch = None
        if self.include_dsm_branch:
            self.dsm_branch = HighResElevationBranch(
                in_channels=1,
                start_channels=elevation_branch_start_channels,
                out_channels=self.elevation_out_channels,
                num_downsample_layers=elevation_branch_downsample_layers,
                kernel_size=elevation_branch_kernel_size
            )
        # --- END NEW ---

        # --- Fusion & Head Projections ---
        self.proj_temporal_output = nn.Conv2d(temporal_output_channels, proj_temporal_ch, kernel_size=1)

        # --- U-Net Head ---
        head_in_ch = proj_temporal_ch
        if self.proj_non_elev_static:
             head_in_ch += proj_static_ch
        if self.include_dem_branch:
             head_in_ch += self.elevation_out_channels
        if self.include_dsm_branch:
             head_in_ch += self.elevation_out_channels

        features = unet_base_channels
        self.unet_depth = unet_depth

        self.unet_inc = UNetConvBlock(head_in_ch, features)
        self.unet_downs = nn.ModuleList()
        self.unet_ups = nn.ModuleList()

        # Downsampling Path
        current_channels = features
        for _ in range(unet_depth):
            self.unet_downs.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    UNetConvBlock(current_channels, current_channels * 2)
                )
            )
            current_channels *= 2

        # Upsampling Path (Using Interpolate + Conv Block)
        for _ in range(unet_depth):
            self.unet_ups.append(
                 UNetUpBlockInterpolate(current_channels, current_channels // 2, scale_factor=2)
            )
            current_channels //= 2

        # Final 1x1 convolution
        self.unet_outc = nn.Conv2d(features, 1, kernel_size=1)

        logging.info(f"BranchedUHIModel (ConvLSTM) initialized:")
        logging.info(f"  Weather Branch (ConvLSTM): Input Ch={weather_input_channels}, Hidden={convlstm_hidden_dims}, Kernels={convlstm_kernel_sizes}, Layers={convlstm_num_layers} -> Proj Ch={proj_temporal_ch}")
        logging.info(f"  Static Branch: Input Ch (Raw Static Non-Elev/Clay)={static_channels}")
        if self.include_clay_features:
            logging.info(f"    -> Including Clay Features: Embed Dim={self.clay_embed_dim}")
            logging.info(f"    -> Clay Backbone Frozen: {freeze_clay_backbone}")
        logging.info(f"  Combined NON-ELEV Static Input Ch: {non_elevation_static_input_ch} -> Proj Ch={proj_static_ch if self.proj_non_elev_static else 'N/A'}")
        logging.info(f"  DEM Branch Included: {self.include_dem_branch} (Output Ch: {self.elevation_out_channels if self.include_dem_branch else 'N/A'}) ")
        logging.info(f"  DSM Branch Included: {self.include_dsm_branch} (Output Ch: {self.elevation_out_channels if self.include_dsm_branch else 'N/A'}) ")
        logging.info(f"  UNet Head: Input Ch={head_in_ch}, Base Ch={unet_base_channels}, Depth={unet_depth}")

    def forward(self,
                weather_seq: torch.Tensor,      # (B, T, C_weather, H, W)
                # --- UPDATED STATIC INPUT --- #
                static_features: Optional[torch.Tensor] = None, # (B, C_static_raw, H, W) - Non-Clay, Non-Elev Static
                cloudless_mosaic: Optional[torch.Tensor] = None, # (B, C_clay, H, W)
                norm_time_tensor: Optional[torch.Tensor] = None, # (B, 4)
                norm_latlon_tensor: Optional[torch.Tensor] = None, # (B, 4)
                # --- NEW: HIGH-RES INPUTS --- #
                high_res_dem: Optional[torch.Tensor] = None, # (B, 1, H_high, W_high)
                high_res_dsm: Optional[torch.Tensor] = None, # (B, 1, H_high, W_high)
                # ---------------------------- #
                # --- Target Size --- #
                target_h_w: Optional[Tuple[int, int]] = None # Target output spatial size (H_orig, W_orig)
                ) -> torch.Tensor:
        """
        Forward pass using ConvLSTM for weather and optional high-res elevation branches.

        Args:
            weather_seq (torch.Tensor): Sequence of weather grids (B, T, C_in, H, W).
            static_features (Optional[torch.Tensor]): Concatenated non-Clay, non-elevation static features (B, C_stat_raw, H, W).
            cloudless_mosaic (Optional[torch.Tensor]): Input for Clay branch (B, C_clay, H, W).
            norm_time_tensor (Optional[torch.Tensor]): Normalized time for Clay (B, 4).
            norm_latlon_tensor (Optional[torch.Tensor]): Normalized lat/lon for Clay (B, 4).
            high_res_dem (Optional[torch.Tensor]): High-resolution DEM map (B, 1, H_high, W_high).
                                                 Provide only if self.include_dem_branch is True.
            high_res_dsm (Optional[torch.Tensor]): High-resolution DSM map (B, 1, H_high, W_high).
                                                 Provide only if self.include_dsm_branch is True.
            target_h_w (Optional[Tuple[int, int]]): Target output spatial size. If None, output size matches feature map size.

        Returns:
            torch.Tensor: Predicted UHI map (B, 1, H_out, W_out).
        """
        B, T, C_w, H, W = weather_seq.shape

        # --- Input Checks --- #
        if self.include_dem_branch and high_res_dem is None:
            raise ValueError("`high_res_dem` must be provided when `include_dem_branch` is True.")
        if self.include_dsm_branch and high_res_dsm is None:
            raise ValueError("`high_res_dsm` must be provided when `include_dsm_branch` is True.")
        # ------------------- #

        # --- Weather Branch ---
        # 1. Optional spatial feature extraction per timestep
        # Reshape for 2D conv: (B, T, C, H, W) -> (B*T, C, H, W)
        weather_seq_flat = weather_seq.view(B * T, C_w, H, W)
        weather_features_flat = self.weather_feature_extractor(weather_seq_flat)
        # Reshape back: (B*T, C_feat, H_feat, W_feat) -> (B, T, C_feat, H_feat, W_feat)
        _, C_w_feat, H_feat, W_feat = weather_features_flat.shape
        weather_features_seq = weather_features_flat.view(B, T, C_w_feat, H_feat, W_feat)

        # 2. Apply ConvLSTM
        # Input: (B, T, C_w_feat, H_feat, W_feat)
        layer_output_seq, last_state = self.convlstm(weather_features_seq)
        # We need the output features of the *last* time step from the *last* layer
        # layer_output_seq is the output of the last layer: (B, T, C_hidden_last, H_feat, W_feat)
        convlstm_last_step_features = layer_output_seq[:, -1, :, :, :] # (B, C_hidden_last, H_feat, W_feat)
        _, _, H_feat, W_feat = convlstm_last_step_features.shape # Get feature map size

        # 3. Project ConvLSTM output
        projected_temporal = self.proj_temporal_output(convlstm_last_step_features)

        # --- Static Branch (Non-Elevation) ---
        all_static_inputs_resized = []
        # 1. Add non-Clay, non-Elev static features (if provided)
        if static_features is not None:
            if static_features.shape[2:] != (H_feat, W_feat):
                static_features_resized = F.interpolate(static_features, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            else:
                static_features_resized = static_features
            all_static_inputs_resized.append(static_features_resized)
        elif not self.include_clay_features: # If no static and no clay, raise error or warning
             logging.warning("No static features (non-Clay or Clay) provided to the model.")
             # Create a dummy zero tensor for projection? Or handle in U-Net?
             # For now, continue, proj_non_elev_static might handle 0 channels if initialized correctly

        # 2. Get and add Clay features if included
        if self.include_clay_features:
            if cloudless_mosaic is None or norm_time_tensor is None or norm_latlon_tensor is None:
                raise ValueError("cloudless_mosaic, norm_time, norm_latlon required for Clay branch.")
            # Ensure Clay backbone is in correct mode (eval or train)
            # self.clay_backbone.model.train(not self.clay_backbone.freeze_backbone) # Handled in Clay init
            with torch.set_grad_enabled(self.clay_backbone.model.training):
                clay_features_native = self.clay_backbone(cloudless_mosaic, norm_time_tensor, norm_latlon_tensor)

            if clay_features_native.shape[2:] != (H_feat, W_feat):
                 clay_features_resized = F.interpolate(clay_features_native, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            else:
                 clay_features_resized = clay_features_native
            all_static_inputs_resized.append(clay_features_resized)

        # 3. Concatenate and Project NON-ELEVATION Static Features
        projected_non_elev_static = None
        if all_static_inputs_resized and self.proj_non_elev_static:
            combined_non_elev_static = torch.cat(all_static_inputs_resized, dim=1)
            projected_non_elev_static = self.proj_non_elev_static(combined_non_elev_static)
        elif not all_static_inputs_resized:
             logging.warning("No non-Clay or Clay static features were provided or enabled.")

        # --- Process High-Res Elevation Branches ---
        dem_features_resized = None
        if self.include_dem_branch and self.dem_branch is not None:
            dem_features = self.dem_branch(high_res_dem) # (B, C_elev_out, H_elev_feat, W_elev_feat)
            # Resize to match ConvLSTM feature map size
            dem_features_resized = F.interpolate(dem_features, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        dsm_features_resized = None
        if self.include_dsm_branch and self.dsm_branch is not None:
            dsm_features = self.dsm_branch(high_res_dsm) # (B, C_elev_out, H_elev_feat, W_elev_feat)
            # Resize to match ConvLSTM feature map size
            dsm_features_resized = F.interpolate(dsm_features, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        # --- Fusion ---
        fused_features_list = [projected_temporal]
        # Add projected non-elevation static features (if they exist)
        if projected_non_elev_static is not None:
            fused_features_list.append(projected_non_elev_static)
        # Add resized DEM features (if they exist)
        if dem_features_resized is not None:
            fused_features_list.append(dem_features_resized)
        # Add resized DSM features (if they exist)
        if dsm_features_resized is not None:
            fused_features_list.append(dsm_features_resized)

        fused_features = torch.cat(fused_features_list, dim=1)

        # --- U-Net Head ---
        skip_connections = []
        unet_out = self.unet_inc(fused_features)
        skip_connections.append(unet_out)
        for i in range(self.unet_depth):
            unet_out = self.unet_downs[i](unet_out)
            if i < self.unet_depth - 1:
                skip_connections.append(unet_out)
        for i in range(self.unet_depth):
            skip = skip_connections.pop()
            unet_out = self.unet_ups[i](unet_out, skip)
        prediction_head_res = self.unet_outc(unet_out) # (B, 1, H_feat, W_feat)

        # --- Final Resizing (Optional) ---
        if target_h_w is not None and prediction_head_res.shape[2:] != target_h_w:
            prediction_final = F.interpolate(prediction_head_res, size=target_h_w, mode='bilinear', align_corners=False)
        else:
            prediction_final = prediction_head_res # (B, 1, H_out, W_out)

        return prediction_final 