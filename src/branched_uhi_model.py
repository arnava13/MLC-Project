#!/usr/bin/env python3
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.model import UNetConvBlock, ClayFeatureExtractor, UNetUpBlock, UNetDecoder, UNetDecoderWithTargetResize, FinalUpsamplerAndProjection, SimpleCNNFeatureHead
from src.ingest.data_utils import determine_target_grid_size, calculate_actual_weather_channels, CANONICAL_WEATHER_FEATURE_ORDER

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

        # Return behavior based on return_all_layers
        if self.return_all_layers:
            # Return the list of hidden state sequences from ALL layers,
            # and the list of last_state tuples from ALL layers
            return layer_output_list, last_state_list
        else:
            # Original behavior: Return only the final state tuple (h, c) of the LAST layer
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

# -----------------------------------------------------------------------------
# Branched UHI Model (MODIFIED) -----------------------------------------------
# -----------------------------------------------------------------------------

class BranchedUHIModel(nn.Module):
    """
    UHI prediction model with separate branches for temporal (weather) and
    static features, fused before a selectable head, followed by final processing.
    """
    def __init__(self,
                 weather_input_channels: int,
                 convlstm_hidden_dims: List[int],
                 convlstm_kernel_sizes: List[Tuple[int, int]],
                 convlstm_num_layers: int,
                 feature_flags: Dict[str, bool],
                 proj_static_ch: int,
                 proj_temporal_ch: int,
                 uhi_grid_resolution_m: int,
                 bounds: List[float],
                 head_type: str = "unet", # "unet" or "simple_cnn"
                 # U-Net Head Params
                 unet_base_channels: int = 32,
                 unet_depth: int = 3,
                 # SimpleCNN Head Params
                 simple_cnn_hidden_dims: Optional[List[int]] = None, # e.g. [64, 32]
                 simple_cnn_output_channels: int = 16, # Output from SimpleCNNFeatureHead
                 simple_cnn_kernel_size: int = 3,
                 simple_cnn_dropout_rate: float = 0.1,
                 # Final Processor Params
                 final_processor_refinement_channels: int = 16, 
                 weather_seq_length: int = 60,
                 sentinel_bands_to_load: Optional[List[str]] = None,
                 enabled_weather_features: Optional[List[str]] = None,
                 clay_model_size: Optional[str] = None,
                 clay_bands: Optional[List[str]] = None,
                 clay_platform: Optional[str] = None,
                 clay_gsd: Optional[int] = None,
                 freeze_backbone: bool = True,
                 clay_checkpoint_path: Optional[Union[str, Path]] = None,
                 clay_metadata_path: Optional[Union[str, Path]] = None,
                 clay_proj_channels: Optional[int] = 32):
        super().__init__()
        self.feature_flags = feature_flags
        self.bounds = bounds
        self.uhi_grid_resolution_m = uhi_grid_resolution_m
        self.head_type = head_type.lower()

        if enabled_weather_features is None: # Default to all if not provided
            logging.warning("'enabled_weather_features' not provided to BranchedUHIModel, defaulting to all canonical. This may not match dataloader!")
            temp_base_features = set()
            for f_name in CANONICAL_WEATHER_FEATURE_ORDER:
                if "_sin" in f_name or "_cos" in f_name:
                    temp_base_features.add(f_name.split('_')[0] + "_" + f_name.split('_')[1])
                else:
                    temp_base_features.add(f_name)
            self.enabled_weather_features = list(temp_base_features)
        else:
            self.enabled_weather_features = enabled_weather_features

        actual_weather_input_channels = calculate_actual_weather_channels(self.enabled_weather_features)

        self.target_H, self.target_W = determine_target_grid_size(
            self.bounds, self.uhi_grid_resolution_m
        )
        logging.info(f"BranchedModel configured for target output UHI grid: ({self.target_H}, {self.target_W})")

        # --- Weather Branch (ConvLSTM) --- (Logic remains same)
        self.conv_lstm = ConvLSTM(
            input_dim=actual_weather_input_channels, hidden_dim=convlstm_hidden_dims,
            kernel_size=convlstm_kernel_sizes, num_layers=convlstm_num_layers,
            batch_first=True, bias=True, return_all_layers=True)
        self.weather_seq_length = weather_seq_length
        self.timestep_weights = nn.Parameter(torch.randn(self.weather_seq_length))

        # --- Static Feature Processing & Clay Backbone --- (Logic remains same)
        self.clay_model = None
        clay_output_channels = 0
        if self.feature_flags.get("use_clay", False):
            if not all([clay_checkpoint_path, clay_metadata_path, clay_model_size, clay_bands, clay_platform, clay_gsd]):
                 raise ValueError("Missing Clay params")
            self.clay_model = ClayFeatureExtractor(
                model_size=clay_model_size, bands=clay_bands, platform=clay_platform, gsd=clay_gsd,
                freeze_backbone=freeze_backbone, checkpoint_path=clay_checkpoint_path, metadata_path=clay_metadata_path)
            clay_raw_channels = self.clay_model.output_channels
            self.clay_proj_dim = clay_proj_channels
            self.clay_proj = nn.Conv2d(clay_raw_channels, self.clay_proj_dim, kernel_size=1, bias=False)
            clay_output_channels = self.clay_proj_dim
            logging.info(f"Added Clay projection Conv1x1: {clay_raw_channels} -> {self.clay_proj_dim} channels")
        else:
            self.clay_model = None
            clay_output_channels = 0
        
        dataloader_static_channels = 0
        if self.feature_flags.get("use_lst", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_dem", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_dsm", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_ndvi", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_ndbi", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_ndwi", False): dataloader_static_channels += 1
        if self.feature_flags.get("use_sentinel_composite", False) and sentinel_bands_to_load:
            dataloader_static_channels += len(sentinel_bands_to_load)

        static_input_channels_to_proj = clay_output_channels + dataloader_static_channels
        logging.info(f"Total input channels for STATIC projection: {static_input_channels_to_proj}")

        # Ensure weather input channels for ConvLSTM is logged correctly
        logging.info(f"ConvLSTM input channels (actual_weather_input_channels): {actual_weather_input_channels}")

        if static_input_channels_to_proj > 0:
            self.static_proj = nn.Conv2d(static_input_channels_to_proj, proj_static_ch, kernel_size=1)
        else:
            self.static_proj = None; proj_static_ch = 0

        temporal_input_channels_to_proj = convlstm_hidden_dims[-1]
        self.temporal_proj = nn.Conv2d(temporal_input_channels_to_proj, proj_temporal_ch, kernel_size=1)

        # --- Instantiate Selected Feature Head --- #
        input_channels_to_head = proj_static_ch + proj_temporal_ch
        if input_channels_to_head == 0: raise ValueError("No features for head.")
        
        channels_from_feature_head = 0
        if self.head_type == "unet":
            self.feature_head = UNetDecoder( # From src.model
                in_channels=input_channels_to_head,
                base_channels=unet_base_channels,
                depth=unet_depth
            )
            channels_from_feature_head = unet_base_channels
            logging.info(f"BranchedModel using UNetDecoder head. Output channels: {channels_from_feature_head}")
        elif self.head_type == "simple_cnn":
            if simple_cnn_hidden_dims is None:
                simple_cnn_hidden_dims = [max(simple_cnn_output_channels * 2, 32), simple_cnn_output_channels]
            self.feature_head = SimpleCNNFeatureHead( # From src.model
                in_channels=input_channels_to_head,
                hidden_dims=simple_cnn_hidden_dims,
                output_channels_head=simple_cnn_output_channels,
                kernel_size=simple_cnn_kernel_size,
                dropout_rate=simple_cnn_dropout_rate
            )
            channels_from_feature_head = simple_cnn_output_channels
            logging.info(f"BranchedModel using SimpleCNNFeatureHead. Output channels: {channels_from_feature_head}")
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        # --- Final Processor --- #
        self.final_processor = FinalUpsamplerAndProjection( # From src.model
            in_channels=channels_from_feature_head,
            refinement_channels=final_processor_refinement_channels,
            target_h=self.target_H,
            target_w=self.target_W
        )
        logging.info(f"BranchedUHIModel initialized with {self.head_type} head and FinalUpsamplerAndProjection.")

    def forward(self, weather_seq: torch.Tensor,
                static_features: Optional[torch.Tensor] = None,
                clay_mosaic: Optional[torch.Tensor] = None,
                norm_latlon: Optional[torch.Tensor] = None,
                norm_timestamp: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # --- 1. Temporal Branch (ConvLSTM) --- (Logic remains same for processing sequence)
        layer_outputs_list, _ = self.conv_lstm(weather_seq)
        temporal_feature_sequence = layer_outputs_list[-1]
        B, T_actual, C_lstm, H_feat, W_feat = temporal_feature_sequence.shape
        if self.timestep_weights.shape[0] != T_actual: raise ValueError("Timestep_weights vs T_actual mismatch")
        normalized_attention_weights = F.softmax(self.timestep_weights, dim=0)
        weights_reshaped = normalized_attention_weights.view(1, T_actual, 1, 1, 1)
        temporal_features_pooled = (temporal_feature_sequence * weights_reshaped).sum(dim=1)
        temporal_projected = self.temporal_proj(temporal_features_pooled)

        # --- 2. Static Branch --- (Logic remains same for feature extraction & projection)
        all_static_features_list = []
        if self.clay_model is not None:
            if clay_mosaic is None or norm_latlon is None or norm_timestamp is None: raise ValueError("Clay inputs missing")
            clay_features_raw = self.clay_model(clay_mosaic, norm_latlon, norm_timestamp)
            if clay_features_raw.shape[-2:] != (H_feat, W_feat):
                clay_features_raw = F.interpolate(clay_features_raw, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            all_static_features_list.append(clay_features_raw)
        if static_features is not None:
             if static_features.shape[-2:] != (H_feat, W_feat): raise ValueError("Static features spatial dim mismatch")
             all_static_features_list.append(static_features)

        static_projected = torch.zeros(B, 0, H_feat, W_feat, device=temporal_projected.device)
        if self.static_proj is not None and all_static_features_list:
            combined_static = torch.cat(all_static_features_list, dim=1)
            if self.static_proj.weight.shape[1] != combined_static.shape[1]: 
                raise ValueError(f"Static proj channel mismatch: {self.static_proj.weight.shape[1]} vs {combined_static.shape[1]}")
            static_projected = self.static_proj(combined_static)
        elif self.static_proj is not None and not all_static_features_list:
             # This means static_proj was created but no static inputs were actually fed to forward
             # This case implies proj_static_ch > 0. Outputting zeros of proj_static_ch.
             static_projected = torch.zeros(B, self.static_proj.out_channels, H_feat, W_feat, device=temporal_projected.device)

        # --- 3. Fusion --- #
        fused_features = torch.cat([static_projected, temporal_projected], dim=1)

        # --- 4. Pass through selected Feature Head --- #
        head_output_features = self.feature_head(fused_features)
        
        # --- 5. Pass through Final Processor --- #
        prediction = self.final_processor(head_output_features)

        return prediction 