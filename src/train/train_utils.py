import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
import shutil
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List, Union, Any
import torch.nn.functional as F # For interpolation
import tempfile # Added for safe saving
from sklearn.metrics import r2_score # <<< Add import for r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_checkpoint(state: Dict[str, Any], is_best: bool, output_dir: Union[str, Path], filename: str = 'checkpoint.pth.tar', best_filename: str = 'model_best.pth.tar'):
    """Saves model checkpoint safely using a temporary file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    best_filepath = output_dir / best_filename

    # Use tempfile for safer saving
    tmp_file_path = None # Initialize path variable
    try:
        # Create a temporary file in the same directory to ensure atomic rename works
        with tempfile.NamedTemporaryFile(delete=False, dir=output_dir, suffix='.tmp') as tmp_file_obj:
            tmp_file_path = Path(tmp_file_obj.name)
            torch.save(state, tmp_file_path)
            # logging.debug(f"Successfully saved state to temporary file {tmp_file_path}") # DEBUG

        # Rename the temporary file to the final filename (atomic on most systems)
        tmp_file_path.rename(filepath)
        logging.info(f"Saved current checkpoint to {filepath}")

        # If this is the best model, copy the saved checkpoint (which now definitely exists)
        if is_best:
            shutil.copyfile(filepath, best_filepath)
            logging.info(f"Saved new best model to {best_filepath}")

    except Exception as e:
        logging.error(f"Error during checkpoint saving process for {filepath}: {e}", exc_info=True)
        # Clean up temporary file if it exists and rename failed or an error occurred after save
        if tmp_file_path is not None and tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
                logging.info(f"Cleaned up partially saved temporary file {tmp_file_path}")
            except OSError as unlink_e:
                logging.error(f"Failed to clean up temporary file {tmp_file_path}: {unlink_e}")
    # No finally block needed as rename handles the tmp file removal implicitly on success


def check_path(path_obj: Path | str, project_root: Path, description: str, should_exist: bool = True, is_absolute: bool = False) -> Path:
    """
    Checks if a given path exists and returns its absolute path.
    If is_absolute is True, path_obj is treated as already absolute or resolvable from CWD.
    Otherwise, it's resolved relative to project_root.
    Converts input to Path object if it's a string.
    """
    if not isinstance(path_obj, Path):
        path_obj = Path(path_obj)

    if is_absolute:
        # If path_obj is already absolute, resolve() cleans it (e.g., resolves '..').
        # If path_obj is relative but is_absolute=True, it's resolved from CWD.
        # This is typically used when the path is constructed to be absolute already.
        abs_path = path_obj.resolve()
    else:
        # This branch is for when path_obj is intended to be relative to project_root
        abs_path = (project_root / path_obj).resolve()

    if should_exist and not abs_path.exists():
        logging.error(f"{description} not found at {abs_path}. Ensure prerequisites are met (e.g., download_data.ipynb ran or file exists).")
        raise FileNotFoundError(f"{description} not found at {abs_path}. Ensure prerequisites are met (e.g., download_data.ipynb ran or file exists).")
    elif not should_exist and abs_path.exists():
        # This case is mainly for output directories or files that might be overwritten.
        logging.info(f"{description} path: {abs_path}. It will be used. If it is an output path, it might be overwritten/appended to.")
    elif not should_exist and not abs_path.exists():
        logging.info(f"{description} path: {abs_path}. It does not exist yet (expected for output paths).")
    # If should_exist is True and it exists, no message needed, just return.
    return abs_path


def split_data(dataset: Dataset, val_percent: float = 0.40, seed: int = 42) -> Tuple[Subset, Optional[Subset]]:
    """Splits a dataset into training and validation subsets."""
    n_samples = len(dataset)
    if n_samples == 0:
        logging.warning("Dataset is empty. Cannot create splits.")
        return dataset, None # Return empty dataset as train, None as val

    if n_samples < 10: # Handle very small datasets
        logging.warning(f"Warning: Dataset size ({n_samples}) is very small. Using all data for training.")
        n_val = 0
        n_train = n_samples
        train_ds = dataset # Use the whole dataset for training
        val_ds = None      # No validation set
    else:
        n_val = int(n_samples * val_percent)
        n_train = n_samples - n_val
        if n_val == 0: # Ensure validation set has at least one sample if possible
             if n_samples > 1:
                 n_val = 1
                 n_train = n_samples - 1
             else: # Only 1 sample, use for training
                 logging.warning("Only 1 sample in dataset, using for training only.")
                 train_ds = dataset
                 val_ds = None
                 logging.info(f"Random dataset split: {n_train} training, 0 validation samples.")
                 return train_ds, val_ds

        # Use random_split for a random split
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)) # Seed for reproducibility

    logging.info(f"Random dataset split: {len(train_ds)} training, {len(val_ds) if val_ds else 0} validation samples.")
    return train_ds, val_ds


def calculate_uhi_stats(train_ds: Subset) -> Tuple[float, float]:
    """Calculates the mean and standard deviation of UHI values from the training subset (memory efficient)."""
    logging.info("Calculating UHI statistics from training data...")
    all_train_targets = []

    if len(train_ds) == 0:
        logging.warning("Training dataset is empty, cannot calculate UHI stats. Returning 0.0, 1.0")
        return 0.0, 1.0

    original_dataset = train_ds.dataset
    indices_to_process = train_ds.indices
    # Check if the original dataset has the required precomputed grids
    if not hasattr(original_dataset, 'unique_timestamps') or \
        not hasattr(original_dataset, 'target_grids') or \
        not hasattr(original_dataset, 'valid_masks'):
        raise AttributeError("Original dataset within Subset does not have required attributes (unique_timestamps, target_grids, valid_masks) for stats calculation.")


    # Iterate through only the required indices
    for idx in tqdm(indices_to_process, desc="Calculating stats"):
        # Directly retrieve only target and mask for this index
        try:
            target_timestamp = original_dataset.unique_timestamps[idx]
            target_grid = original_dataset.target_grids[target_timestamp]
            valid_mask = original_dataset.valid_masks[target_timestamp]
            target_tensor = torch.from_numpy(target_grid).float()
            mask_tensor = torch.from_numpy(valid_mask).bool()
        except IndexError:
             logging.error(f"Index {idx} out of bounds for unique_timestamps (len {len(original_dataset.unique_timestamps)}). Skipping.")
             continue
        except KeyError:
             logging.error(f"Timestamp {target_timestamp} not found in precomputed target_grids or valid_masks. Skipping index {idx}.")
             continue
        except Exception as e:
             logging.error(f"Error accessing data for index {idx}: {e}. Skipping.")
             continue


        # Apply mask and collect valid UHI values
        valid_targets = target_tensor[mask_tensor] # Mask is already boolean
        if valid_targets.numel() > 0: # Check if any valid targets exist
            all_train_targets.append(valid_targets.cpu()) # Move to CPU before collecting

    uhi_mean = 0.0
    uhi_std = 1.0 # Default standard deviation

    if not all_train_targets:
         logging.warning("No valid training targets found after iterating. Check masks or data. Returning default stats (0.0, 1.0).")
    else:
        all_train_targets_tensor = torch.cat(all_train_targets)
        if all_train_targets_tensor.numel() > 0:
            uhi_mean = all_train_targets_tensor.mean().item()
            uhi_std_calc = all_train_targets_tensor.std().item()
            # Add a small epsilon to std to prevent division by zero if all targets are identical
            uhi_std = uhi_std_calc if uhi_std_calc > 1e-6 else 1.0
        else:
            logging.warning("Concatenated target tensor is empty. Returning default stats (0.0, 1.0).")


    logging.info(f"Training UHI Mean: {uhi_mean:.4f}, Std Dev: {uhi_std:.4f}")
    return uhi_mean, uhi_std


def create_dataloaders(train_ds: Subset,
                         val_ds: Optional[Subset],
                         n_train_batches: int,
                         num_workers: int,
                         device: torch.device) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Creates training and validation dataloaders."""
    logging.info("Creating dataloaders...")
    train_loader = None
    val_loader = None

    pin_memory = device.type == 'cuda'

    # Calculate train batch size
    if train_ds and len(train_ds) > 0:
        if n_train_batches > 0:
             train_batch_size = max(1, len(train_ds) // n_train_batches)
        else:
             logging.warning("n_train_batches is 0 or less. Defaulting train batch size to 1.")
             train_batch_size = 1
        logging.info(f"Using Train Batch Size: {train_batch_size}")
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        logging.warning("Training dataset is empty or None, train_loader not created.")


    # Calculate val batch size (use full validation set if possible)
    if val_ds and len(val_ds) > 0:
        val_batch_size = 1 # Decrease validation batch size to 1
        logging.info(f"Using Validation Batch Size: {val_batch_size}")
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        logging.warning("Validation dataset is empty or None, val_loader not created.")

    logging.info("Data loading setup complete.")
    return train_loader, val_loader 


def validate_model(model, val_loader, loss_fn, device=None):
    """Evaluates model performance on a validation set, properly handling NaN values."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_metrics = {"mae": 0.0, "mse": 0.0, "valid_pixels": 0}
    
    with torch.no_grad():
        for batch in val_loader:
            # Move necessary inputs to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'mask' and k != 'target'}
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            output = model(**inputs)
            
            # Check for NaNs in the model output
            if torch.isnan(output).any():
                print(f"Warning: NaN values detected in model output")
                # Replace NaNs with zeros (or any other appropriate value) for loss calculation
                output = torch.nan_to_num(output, nan=0.0)
            
            # Make sure mask excludes NaN values in target
            if torch.isnan(target).any():
                nan_mask = torch.isnan(target)
                mask = mask & (~nan_mask)  # Only consider valid pixels that are not NaN
            
            # Calculate loss
            loss = loss_fn(output, target, mask)
            
            # Calculate additional metrics
            valid_mask = mask.bool()
            if valid_mask.any():
                pred_valid = output[valid_mask]
                target_valid = target[valid_mask]
                
                # MAE and MSE over valid pixels
                mae = torch.abs(pred_valid - target_valid).mean().item()
                mse = torch.pow(pred_valid - target_valid, 2).mean().item()
                
                all_metrics["mae"] += mae
                all_metrics["mse"] += mse
                all_metrics["valid_pixels"] += valid_mask.sum().item()
            
            total_loss += loss.item()
            num_batches += 1
    
    # Compute average metrics
    avg_metrics = {}
    if num_batches > 0:
        avg_metrics["loss"] = total_loss / num_batches
        avg_metrics["mae"] = all_metrics["mae"] / num_batches
        avg_metrics["mse"] = all_metrics["mse"] / num_batches
        avg_metrics["rmse"] = (all_metrics["mse"] / num_batches) ** 0.5
        avg_metrics["valid_pixels_per_batch"] = all_metrics["valid_pixels"] / num_batches
    
    return avg_metrics


def train_epoch_generic(model: nn.Module,
                          dataloader: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          loss_fn: callable,
                          device: torch.device,
                          uhi_mean: float,
                          uhi_std: float,
                          feature_flags: Dict[str, bool],
                          max_grad_norm: float = 1.0,
                          desc: str = 'Training') -> Tuple[float, float, float]:
    """Trains a generic UHI model for one epoch, handling different batch structures robustly."""
    model.train()
    total_loss = 0.0
    all_targets_unnorm = []
    all_preds_unnorm = []
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    for batch_idx, batch in enumerate(progress_bar):

        # --- Explicit Tensor Extraction --- # 
        target = batch.get('target')
        mask = batch.get('mask')
        weather = batch.get('weather') # For CNN
        weather_seq = batch.get('weather_seq') # For Branched
        input_temporal_seq = batch.get('input_temporal_seq') # NEW: For autoregressive Branched
        static_features = batch.get('static_features')

        # Clay-specific inputs
        clay_mosaic_tensor = None
        norm_latlon_tensor = None
        norm_timestamp_tensor = None

        if feature_flags.get("use_clay", False):
            clay_mosaic_tensor = batch.get('clay_mosaic') 
            norm_latlon_tensor = batch.get('norm_latlon')
            norm_timestamp_tensor = batch.get('norm_timestamp')

            if clay_mosaic_tensor is None or norm_latlon_tensor is None or norm_timestamp_tensor is None:
                logging.error(f"Batch {batch_idx}: 'use_clay' is True but one or more Clay inputs ('clay_mosaic', 'norm_latlon', 'norm_timestamp') not found in batch. Skipping batch.")
                progress_bar.set_postfix(loss=float('nan'))
                continue # Skip this batch

        # --- Basic Checks and Move to Device --- # 
        if target is None or mask is None:
            logging.warning(f"Batch {batch_idx}: Missing 'target' or 'mask'. Skipping batch.")
            continue
        target = target.to(device)
        mask = mask.to(device)

        # Normalize the target tensor using uhi_mean and uhi_std
        target_normalized = (target - uhi_mean) / (uhi_std + 1e-10) # Add epsilon for stability
        
        # Move other tensors if they exist
        weather_input = None # Keep track of the primary temporal input
        model_args = {}
        if input_temporal_seq is not None:
             model_args['input_temporal_seq'] = input_temporal_seq.to(device)
             weather_input = model_args['input_temporal_seq'] # Use this as primary
        elif weather_seq is not None:
             # Fallback for non-autoregressive branched?
             logging.warning(f"Batch {batch_idx}: Found 'weather_seq' but not 'input_temporal_seq'. Using 'weather_seq'.")
             model_args['weather_seq'] = weather_seq.to(device)
             weather_input = model_args['weather_seq']
        elif weather is not None:
             # Fallback for CNN
             model_args['weather'] = weather.to(device)
             weather_input = model_args['weather']
        else:
            logging.error(f"Batch {batch_idx}: Missing primary temporal input ('input_temporal_seq', 'weather_seq', or 'weather'). Skipping batch.")
            continue 
            
        if static_features is not None: model_args['static_features'] = static_features.to(device)
        
        # --- ADD CLAY INPUTS TO MODEL_ARGS IF ENABLED ---
        if feature_flags.get("use_clay", False):
            if clay_mosaic_tensor is not None: model_args['clay_mosaic'] = clay_mosaic_tensor.to(device)
            if norm_latlon_tensor is not None: model_args['norm_latlon'] = norm_latlon_tensor.to(device)
            if norm_timestamp_tensor is not None: model_args['norm_timestamp'] = norm_timestamp_tensor.to(device)
            # The check for None on these tensors was already done earlier when extracting from batch
        # --- END ADD CLAY INPUTS ---

        # --- Forward Pass --- #
        optimizer.zero_grad()
        try:
            # Separate temporal sequence (positional) from other args (keyword)
            if 'input_temporal_seq' not in model_args:
                logging.error(f"Batch {batch_idx}: 'input_temporal_seq' missing from model_args before forward pass. Keys: {model_args.keys()}")
                continue
            temporal_input = model_args.pop('input_temporal_seq')
            pred = model(temporal_input, **model_args)
        except TypeError as e:
            logging.error(f"Model forward pass failed (TypeError) on batch {batch_idx}. Input Shapes - Temporal: {temporal_input.shape if 'temporal_input' in locals() else 'N/A'}, Others: { {k: v.shape for k, v in model_args.items()} }. Error: {e}")
            continue # Skip batch if forward call fails
        except Exception as e:
            logging.error(f"Model forward pass failed (Other Error) on batch {batch_idx}. Input Shapes - Temporal: {temporal_input.shape if 'temporal_input' in locals() else 'N/A'}, Others: { {k: v.shape for k, v in model_args.items()} }. Error: {e}")
            continue # Skip batch

        # --- Loss Calculation (model outputs assumed already in normalized scale) --- #
        try:
             loss = loss_fn(pred, target_normalized, mask)
             if not torch.isfinite(loss):
                  logging.warning(f"Batch {batch_idx}: Loss is NaN/Inf ({loss.item()}). Skipping backward pass.")
                  continue
        except Exception as e:
             logging.error(f"Loss calculation failed on batch {batch_idx}: {e}")
             continue

        # ------ Backward pass ------ #
        try:
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        except Exception as e:
            logging.error(f"Backward pass or optimizer step failed on batch {batch_idx}: {e}")
            # Consider if we should continue or stop training on backward errors
            continue 

        # ------ Metrics calculation ------ #
        with torch.no_grad(): # Ensure metrics calc doesn't affect gradients
             # Un-normalise predictions back to raw scale for metrics
             pred_unnorm = (pred * uhi_std + uhi_mean).squeeze(1) if pred.ndim == 4 else (pred * uhi_std + uhi_mean)
             target_unnorm = target.squeeze(1) if target.ndim == 4 else target
             mask_flat = mask.squeeze(1).bool() if mask.ndim == 4 else mask.bool()
             
             # Update running statistics
             total_loss += loss.item()
             all_targets_unnorm.append(target_unnorm[mask_flat].detach().cpu())
             all_preds_unnorm.append(pred_unnorm[mask_flat].detach().cpu())
             num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': total_loss / num_batches if num_batches > 0 else float('nan')})
        
    # --- Epoch Metrics Calculation --- #
    if num_batches == 0:
         logging.warning("No valid batches processed in training epoch.")
         return float('nan'), float('nan'), float('nan')
         
    avg_loss = total_loss / num_batches
    try:
        all_targets_unnorm = torch.cat(all_targets_unnorm, dim=0).flatten()
        all_preds_unnorm = torch.cat(all_preds_unnorm, dim=0).flatten()
        
        # Ensure tensors are valid for metric calculation
        valid_targets = all_targets_unnorm[torch.isfinite(all_targets_unnorm)]
        valid_preds = all_preds_unnorm[torch.isfinite(all_preds_unnorm)]
        # Align predictions and targets based on validity (conservative approach)
        valid_indices = torch.isfinite(all_targets_unnorm) & torch.isfinite(all_preds_unnorm)
        aligned_targets = all_targets_unnorm[valid_indices]
        aligned_preds = all_preds_unnorm[valid_indices]

        if aligned_targets.numel() == 0:
             logging.warning("No valid finite target/prediction pairs for epoch metrics.")
             rmse = float('nan')
             r2 = float('nan')
        else:
             rmse = torch.sqrt(torch.mean((aligned_preds - aligned_targets) ** 2)).item()
             # Use numpy for r2_score as it handles edge cases well
             r2 = r2_score(aligned_targets.numpy(), aligned_preds.numpy())
             
    except Exception as e:
        logging.error(f"Error calculating epoch metrics: {e}")
        rmse = float('nan')
        r2 = float('nan')
        
    return avg_loss, rmse, r2


def validate_epoch_generic(model: nn.Module,
                            dataloader: DataLoader,
                            loss_fn: callable,
                            device: torch.device,
                            uhi_mean: float,
                            uhi_std: float,
                            feature_flags: Dict[str, bool],
                            desc: str = 'Validation') -> Tuple[float, float, float]:
    """Validates a generic UHI model for one epoch, handling different batch structures robustly."""
    model.eval()
    total_loss = 0.0
    all_targets_unnorm = []
    all_preds_unnorm = []
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            
            # --- Explicit Tensor Extraction --- # 
            target = batch.get('target')
            mask = batch.get('mask')
            weather = batch.get('weather') # For CNN
            weather_seq = batch.get('weather_seq') # For Branched
            input_temporal_seq = batch.get('input_temporal_seq') # NEW: For autoregressive Branched
            static_features = batch.get('static_features')

            # Clay-specific inputs
            clay_mosaic_tensor = None
            norm_latlon_tensor = None
            norm_timestamp_tensor = None

            if feature_flags.get("use_clay", False):
                clay_mosaic_tensor = batch.get('clay_mosaic') 
                norm_latlon_tensor = batch.get('norm_latlon')
                norm_timestamp_tensor = batch.get('norm_timestamp')

                if clay_mosaic_tensor is None or norm_latlon_tensor is None or norm_timestamp_tensor is None:
                    logging.warning(f"Epoch {desc}, Batch {batch_idx}: 'use_clay' is True but one or more Clay inputs ('clay_mosaic', 'norm_latlon', 'norm_timestamp') not found in batch. Skipping batch.")
                    progress_bar.set_postfix(loss=float('nan')) # Or some indicator of skip
                    continue # Skip this batch

            # --- Basic Checks and Move to Device --- # 
            if target is None or mask is None:
                logging.warning(f"Validation Batch {batch_idx}: Missing 'target' or 'mask'. Skipping batch.")
                continue
            target = target.to(device)
            mask = mask.to(device)

            # Normalize the target tensor using uhi_mean and uhi_std
            target_normalized = (target - uhi_mean) / (uhi_std + 1e-9) # Add epsilon for stability
            
            # Move other tensors if they exist
            weather_input = None # Keep track of the primary temporal input
            model_args = {}
            if input_temporal_seq is not None:
                 model_args['input_temporal_seq'] = input_temporal_seq.to(device)
                 weather_input = model_args['input_temporal_seq']
            elif weather_seq is not None:
                 logging.warning(f"Validation Batch {batch_idx}: Found 'weather_seq' but not 'input_temporal_seq'. Using 'weather_seq'.")
                 model_args['weather_seq'] = weather_seq.to(device)
                 weather_input = model_args['weather_seq']
            elif weather is not None:
                 model_args['weather'] = weather.to(device)
                 weather_input = model_args['weather']
            else:
                logging.warning(f"Validation Batch {batch_idx}: Missing primary temporal input ('input_temporal_seq', 'weather_seq', or 'weather'). Skipping batch.")
                continue
                
            if static_features is not None: model_args['static_features'] = static_features.to(device)

            # --- ADD CLAY INPUTS TO MODEL_ARGS IF ENABLED ---
            if feature_flags.get("use_clay", False):
                if clay_mosaic_tensor is not None: model_args['clay_mosaic'] = clay_mosaic_tensor.to(device)
                if norm_latlon_tensor is not None: model_args['norm_latlon'] = norm_latlon_tensor.to(device)
                if norm_timestamp_tensor is not None: model_args['norm_timestamp'] = norm_timestamp_tensor.to(device)
                # The check for None on these tensors was already done earlier when extracting from batch
            # --- END ADD CLAY INPUTS ---

            # ------ Forward pass and loss calculation ------ #
            try:
                # Separate temporal sequence (positional) from other args (keyword)
                if 'input_temporal_seq' not in model_args:
                    logging.error(f"Validation Batch {batch_idx}: 'input_temporal_seq' missing from model_args before forward pass. Keys: {model_args.keys()}")
                    continue
                temporal_input = model_args.pop('input_temporal_seq')
                pred = model(temporal_input, **model_args)
                
                if not torch.isfinite(pred).all():
                    logging.warning(f"Validation Batch {batch_idx}: NaN or Inf detected in model predictions! Skipping.")
                    continue
                    
                # --- Loss Calculation (model outputs assumed already in normalized scale) --- #
                loss = loss_fn(pred, target_normalized, mask)
                if not torch.isfinite(loss):
                    logging.warning(f"Validation Batch {batch_idx}: Loss is NaN/Inf ({loss.item()})! Skipping.")
                    continue
                    
            except TypeError as e:
                logging.error(f"Validation model forward pass failed (TypeError) on batch {batch_idx}. Input Shapes - Temporal: {temporal_input.shape if 'temporal_input' in locals() else 'N/A'}, Others: { {k: v.shape for k, v in model_args.items()} }. Error: {e}")
                continue
            except Exception as e:
                logging.error(f"Validation forward/loss failed (Other Error) on batch {batch_idx}. Input Shapes - Temporal: {temporal_input.shape if 'temporal_input' in locals() else 'N/A'}, Others: { {k: v.shape for k, v in model_args.items()} }. Error: {e}")
                continue
            
            # ------ Metrics calculation ------ #
            # Un-normalise predictions back to raw scale for metrics
            pred_unnorm = (pred * uhi_std + uhi_mean).squeeze(1) if pred.ndim == 4 else (pred * uhi_std + uhi_mean)
            target_unnorm = target.squeeze(1) if target.ndim == 4 else target
            mask_flat = mask.squeeze(1).bool() if mask.ndim == 4 else mask.bool()
            
            # Update running statistics
            total_loss += loss.item()
            all_targets_unnorm.append(target_unnorm[mask_flat].detach().cpu())
            all_preds_unnorm.append(pred_unnorm[mask_flat].detach().cpu())
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Val Loss': total_loss / num_batches if num_batches > 0 else float('nan')})
    
    # --- Epoch Metrics Calculation --- #
    if num_batches == 0:
         logging.warning("No valid batches processed in validation epoch.")
         return float('nan'), float('nan'), float('nan')
         
    avg_loss = total_loss / num_batches
    try:
        all_targets_unnorm = torch.cat(all_targets_unnorm, dim=0).flatten()
        all_preds_unnorm = torch.cat(all_preds_unnorm, dim=0).flatten()
        
        # Align predictions and targets based on validity (conservative approach)
        valid_indices = torch.isfinite(all_targets_unnorm) & torch.isfinite(all_preds_unnorm)
        aligned_targets = all_targets_unnorm[valid_indices]
        aligned_preds = all_preds_unnorm[valid_indices]

        if aligned_targets.numel() == 0:
             logging.warning("No valid finite target/prediction pairs for validation epoch metrics.")
             rmse = float('nan')
             r2 = float('nan')
        else:
             rmse = torch.sqrt(torch.mean((aligned_preds - aligned_targets) ** 2)).item()
             r2 = r2_score(aligned_targets.numpy(), aligned_preds.numpy())
             
    except Exception as e:
        logging.error(f"Error calculating validation epoch metrics: {e}")
        rmse = float('nan')
        r2 = float('nan')
    
    return avg_loss, rmse, r2

# --- End Generic Train/Validate Functions --- # 