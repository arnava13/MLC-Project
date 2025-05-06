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


def check_path(relative_path: Optional[Union[str, Path]],
               project_root: Union[str, Path],
               description: str,
               should_exist: bool = True) -> Optional[Path]:
    """Checks if a path exists relative to the project root.

    Args:
        relative_path: Path relative to project root. Can be None.
        project_root: Absolute path to the project root.
        description: Description of the file/directory for error messages.
        should_exist: If True, raises FileNotFoundError if the path doesn't exist.

    Returns:
        Absolute Path object if relative_path is not None, otherwise None.
    """
    if relative_path is None:
        return None
    project_root = Path(project_root)
    abs_path = project_root / relative_path
    if should_exist and not abs_path.exists():
        raise FileNotFoundError(f"{description} not found at {abs_path}. Ensure prerequisites are met (e.g., download_data.ipynb ran).")
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
        val_batch_size = 32 # Use a fixed batch size for validation to manage memory
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


# --- NEW Generic Train/Validate Epoch Functions --- #

def train_epoch_generic(model: nn.Module,
                          dataloader: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          loss_fn: callable,
                          device: torch.device,
                          uhi_mean: float,
                          uhi_std: float,
                          desc: str = 'Training') -> Tuple[float, float, float]:
    """Trains a generic UHI model for one epoch, handling different batch structures."""
    model.train()
    total_loss = 0.0
    all_targets_unnorm = []
    all_preds_unnorm = []
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        # --- Move data to device --- #
        target = batch['target'].to(device)
        mask = batch['mask'].to(device)
        static_features = batch.get('static_features')
        if static_features is not None: static_features = static_features.to(device)
        clay_mosaic = batch.get('cloudless_mosaic')
        if clay_mosaic is not None: clay_mosaic = clay_mosaic.to(device)
        norm_latlon = batch.get('norm_latlon')
        if norm_latlon is not None: norm_latlon = norm_latlon.to(device)
        norm_timestamp = batch.get('norm_timestamp')
        if norm_timestamp is not None: norm_timestamp = norm_timestamp.to(device)

        # --- Determine model-specific inputs --- #
        model_inputs = {
            'static_features': static_features,
            'clay_mosaic': clay_mosaic,
            'norm_latlon': norm_latlon,
            'norm_timestamp': norm_timestamp
        }
        if 'weather' in batch: # For UHINetCNN
            model_inputs['weather'] = batch['weather'].to(device)
        elif 'weather_seq' in batch: # For BranchedUHIModel
            model_inputs['weather_seq'] = batch['weather_seq'].to(device)
        else:
            logging.warning("Batch missing 'weather' or 'weather_seq'. Cannot pass temporal data.")
            # Continue without temporal data if model can handle it, otherwise error

        # --- Forward Pass --- #
        try:
            predictions = model(**model_inputs)
        except TypeError as e:
            logging.error(f"Error during model forward pass: {e}")
            logging.error(f"Model type: {type(model).__name__}, Available keys in batch: {batch.keys()}")
            logging.error(f"Inputs passed to model: {model_inputs.keys()}")
            raise

        # Ensure prediction spatial dims match target/mask for loss calculation
        if predictions.shape[-2:] != target.shape[-2:]:
            predictions = F.interpolate(predictions, size=target.shape[-2:], mode='bilinear', align_corners=False)

        loss = loss_fn(predictions, target, mask)

        # --- Backpropagation --- #
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        # A max_norm of 1.0 is a common value that works well in many cases
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # --- Metrics Calculation --- #
        total_loss += loss.item()

        # Convert tensors to numpy for evaluation metrics (no extra scaling)
        targets_unnorm = target.cpu().numpy()
        preds_unnorm = predictions.detach().cpu().numpy()
        valid_mask_np = mask.cpu().numpy().astype(bool) # Ensure boolean mask

        # Store only valid pixels efficiently
        valid_targets = targets_unnorm[valid_mask_np]
        valid_preds = preds_unnorm[valid_mask_np]

        if valid_targets.size > 0: # Check if there are any valid pixels
            all_targets_unnorm.append(valid_targets)
            all_preds_unnorm.append(valid_preds)

        num_batches += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate overall epoch metrics (RMSE, R2) on all valid pixels
    rmse_epoch = 0.0
    r2_epoch = 0.0
    if all_targets_unnorm:
        all_targets_flat = np.concatenate(all_targets_unnorm)
        all_preds_flat = np.concatenate(all_preds_unnorm)
        if all_targets_flat.size > 0:
            mse = np.mean((all_preds_flat - all_targets_flat)**2)
            rmse_epoch = np.sqrt(mse)
            target_variance = np.var(all_targets_flat)
            epsilon = 1e-10
            r2_epoch = 1 - (mse / (target_variance + epsilon))

    return avg_loss, rmse_epoch, r2_epoch


def validate_epoch_generic(model: nn.Module,
                           dataloader: DataLoader,
                           loss_fn: callable,
                           device: torch.device,
                           uhi_mean: float,
                           uhi_std: float,
                           desc: str = 'Validation') -> Tuple[float, float, float]:
    """Validates a generic UHI model for one epoch, handling different batch structures."""
    model.eval()
    total_loss = 0.0
    all_targets_unnorm = []
    all_preds_unnorm = []
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # --- Move data to device --- #
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            static_features = batch.get('static_features')
            if static_features is not None: static_features = static_features.to(device)
            clay_mosaic = batch.get('cloudless_mosaic')
            if clay_mosaic is not None: clay_mosaic = clay_mosaic.to(device)
            norm_latlon = batch.get('norm_latlon')
            if norm_latlon is not None: norm_latlon = norm_latlon.to(device)
            norm_timestamp = batch.get('norm_timestamp')
            if norm_timestamp is not None: norm_timestamp = norm_timestamp.to(device)

            # --- Determine model-specific inputs --- #
            model_inputs = {
                'static_features': static_features,
                'clay_mosaic': clay_mosaic,
                'norm_latlon': norm_latlon,
                'norm_timestamp': norm_timestamp
            }
            if 'weather' in batch:
                model_inputs['weather'] = batch['weather'].to(device)
            elif 'weather_seq' in batch:
                model_inputs['weather_seq'] = batch['weather_seq'].to(device)
            else:
                logging.warning("Batch missing 'weather' or 'weather_seq' during validation.")

            # --- Forward Pass --- #
            try:
                predictions = model(**model_inputs)
            except TypeError as e:
                logging.error(f"Error during model forward pass (validation): {e}")
                logging.error(f"Model type: {type(model).__name__}, Available keys in batch: {batch.keys()}")
                logging.error(f"Inputs passed to model: {model_inputs.keys()}")
                raise

            # Ensure prediction spatial dims match target/mask for metric calculation
            if predictions.shape[-2:] != target.shape[-2:]:
                predictions = F.interpolate(predictions, size=target.shape[-2:], mode='bilinear', align_corners=False)

            loss = loss_fn(predictions, target, mask)
            total_loss += loss.item()

            # --- Metrics Calculation (raw scale) --- #
            targets_unnorm = target.cpu().numpy()
            preds_unnorm = predictions.cpu().numpy()
            valid_mask_np = mask.cpu().numpy().astype(bool)

            valid_targets = targets_unnorm[valid_mask_np]
            valid_preds = preds_unnorm[valid_mask_np]

            if valid_targets.size > 0:
                all_targets_unnorm.append(valid_targets)
                all_preds_unnorm.append(valid_preds)

            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate overall epoch metrics (RMSE, R2)
    rmse_epoch = 0.0
    r2_epoch = 0.0 # Initialize with a default value
    if all_targets_unnorm:
        all_targets_flat = np.concatenate(all_targets_unnorm)
        all_preds_flat = np.concatenate(all_preds_unnorm)
        if all_targets_flat.size > 0:
            mse = np.mean((all_preds_flat - all_targets_flat)**2)
            rmse_epoch = np.sqrt(mse)
            target_variance = np.var(all_targets_flat)
            
            # Calculate R2 directly, adding epsilon to variance to avoid division by zero
            # Removed specific handling for target_variance <= 1e-6
            epsilon = 1e-9
            r2_epoch = 1 - (mse / (target_variance + epsilon))
            
            # Remove debug print from previous step
            # print(f"[DEBUG R2 VAL] Target Variance: {target_variance:.8e}") 
            # print(f"[DEBUG R2 VAL] Variance <= 1e-6. Predictions close to targets? {np.allclose(all_preds_flat, all_targets_flat)}. Setting R2 to 1.0 if True else 0.0")

    return avg_loss, rmse_epoch, r2_epoch

# --- End Generic Train/Validate Functions --- # 