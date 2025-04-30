1. ConvGRU Implementation:
Since PyTorch doesn't have a built-in ConvGRU, we first need to implement it. We'll add a ConvGRUCell and a ConvGRU layer to src/model.py.
2. Model Architecture (UHINetCNN in src/model.py):
__init__:
Instantiate the ConvGRU layer to process the weather sequence. We need to define the number of hidden channels (cnn_hidden_dims[-1] could be a starting point).
Define the layers for the U-Net decoder head. This head will take the concatenated features (ConvGRU hidden state + projected Clay + optional LST) as input. The input channels will be convgru_hidden_dim + proj_ch (+ lst_channels). It should output 1 channel (the UHI prediction).
forward:
Get static features: projected_clay = self.proj(self.clay_backbone(...)) and potentially resize static_lst. Keep these with shape [B, C, H_feat, W_feat].
Process weather_seq ([B, T, C_weather, H_orig, W_orig]) through the ConvGRU layer. This will output a sequence of hidden states hidden_seq ([B, T, C_hidden, H_orig, W_orig]).
Initialize an empty list to store per-timestep predictions.
Iterate through each timestep t from 0 to T-1:
Get the hidden state for this timestep: h_t = hidden_seq[:, t, :, :, :] ([B, C_hidden, H_orig, W_orig]).
Resize h_t to match the spatial dimensions of static features (H_feat, W_feat): h_t_resized = F.interpolate(h_t, size=(H_feat, W_feat), ...).
Concatenate h_t_resized with the static features (projected_clay and maybe static_lst_resized). Since static features don't change with t, we use the same ones each time. Concatenated shape: [B, C_hidden + C_proj (+ C_lst), H_feat, W_feat].
Pass the concatenated features through the U-Net decoder head to get the prediction for this timestep: pred_t = self.unet_head(concatenated_features_t). Shape: [B, 1, H_target, W_target].
Append pred_t to the list of predictions.
Stack the list of predictions along the time dimension (dim=1) to get the final output sequence: prediction_seq = torch.stack(predictions_list, dim=1). Shape: [B, T, 1, H_target, W_target].
Return prediction_seq.
3. Data Loading (CityDataSet in src/ingest/dataloader.py):
__getitem__: This method needs the most significant change. It currently returns data for a single target timestamp idx. Now, it needs to return data corresponding to the sequence ending at idx.
Identify the sequence of T timestamps ending at idx.
Load or build the weather_seq for these T timestamps ([T, C_weather, H_orig, W_orig]).
Load or build the sequence of target UHI maps for these T timestamps ([T, 1, H_target, W_target]).
Load or build the sequence of mask tensors for these T timestamps ([T, 1, H_target, W_target]).
The cloudless_mosaic and static_lst remain single tensors ([C, H, W]).
norm_time and norm_latlon should still correspond to the first timestamp of the sequence (or potentially the target timestamp idx - we need to be consistent with Clay's expectation, let's assume first timestamp for now).
The returned dictionary should now contain weather_seq, target_seq, mask_seq instead of the single-timestep versions.
4. Training Loop (train_uhi_cnn.ipynb - Cell 8 & 10):
Epoch Functions (train_epoch, validate_epoch):
Unpack batch['target_seq'] and batch['mask_seq'].
The model output will be prediction_seq ([B, T, 1, H_target, W_target]).
Loss Calculation: The masked_mae_loss (or MSE) needs to handle the time dimension. A common approach is to reshape predictions and targets before calculating the loss:
prediction_seq_flat = prediction_seq.view(B * T, 1, H, W)
target_seq_flat = target_seq.view(B * T, 1, H, W)
mask_seq_flat = mask_seq.view(B * T, 1, H, W)
loss = loss_fn(prediction_seq_flat, target_seq_flat, mask_seq_flat, ...)
Metrics: RMSE and R² should also be calculated on the flattened or appropriately aggregated predictions/targets.
Let's start by implementing the ConvGRUCell and ConvGRU and modifying the UHINetCNN structure in src/model.py. We'll handle the dataloader and training loop changes afterward.
I'll first add the ConvGRU components and then modify UHINetCNN. Note: I'll need to define a basic U-Net structure for the head; we can refine it later.