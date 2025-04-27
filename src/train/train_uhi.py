import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd # Needed for loading bounds from csv
from tqdm import tqdm
import json
import os
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.dataloader import CityDataSet
from src.models.uhi_net import UHINetConvGRU
from src.train.loss import masked_mae_loss, masked_mse_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        import shutil
        shutil.copyfile(filename, best_filename)
        logging.info(f"Saved new best model to {best_filename}")

# Training function
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        # Move batch to device
        batch_device = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        predictions = model(batch_device)
        loss = loss_fn(predictions, batch_device['target'], batch_device['mask'])

        # Check for NaN loss
        if torch.isnan(loss):
             logging.warning("NaN loss detected, skipping batch.")
             continue

        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / num_batches if num_batches > 0 else 0.0

# Validation function
def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch_device)
            loss = loss_fn(predictions, batch_device['target'], batch_device['mask'])
            if torch.isnan(loss):
                 logging.warning("NaN validation loss detected, skipping batch.")
                 continue

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / num_batches if num_batches > 0 else 0.0

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading Setup ---
    data_dir_path = Path(args.data_dir)
    city_data_dir = data_dir_path / args.city_name
    uhi_csv = city_data_dir / "uhi_data.csv"
    bbox_csv = city_data_dir / "bbox.csv"
    weather_csv = city_data_dir / "weather_grid.csv"
    cloudless_mosaic_path = Path(args.cloudless_mosaic_path)

    # Determine path for single LST median file if needed
    single_lst_median_file = None
    if args.include_lst:
        # Construct the expected filename based on the convention in create_sat_tensor_files
        # This requires knowing the date range used to generate it.
        # For simplicity, let's require it as an argument for now.
        if not args.single_lst_median_path:
            logging.error("LST is included (--include_lst) but path to the single LST median file (--single_lst_median_path) was not provided.")
            return
        single_lst_median_file = Path(args.single_lst_median_path)
        if not single_lst_median_file.exists():
             logging.error(f"Provided single LST median file not found: {single_lst_median_file}")
             return

    # Basic file checks
    required_files = [uhi_csv, bbox_csv, weather_csv, cloudless_mosaic_path]
    for f in required_files:
        if not f.exists():
            logging.error(f"Required file/directory not found: {f}")
            return
    # LST file existence checked above if include_lst is True

    # Load bounds
    bounds = args.bounds
    if not bounds:
        logging.info("Bounds not provided via args, loading from bbox.csv")
        try:
             bbox_df = pd.read_csv(bbox_csv)
             bounds = [
                 bbox_df['longitudes'].min(), bbox_df['latitudes'].min(),
                 bbox_df['longitudes'].max(), bbox_df['latitudes'].max()
             ]
             logging.info(f"Loaded bounds from {bbox_csv}: {bounds}")
        except Exception as e:
             logging.error(f"Failed to load bounds from {bbox_csv}: {e}. Provide via --bounds.")
             return

    # --- Initialize Dataset ---
    try:
        dataset = CityDataSet(
            bounds=bounds,
            # averaging_window=args.averaging_window, # Not needed for single LST
            resolution_m=args.resolution_m,
            uhi_csv=str(uhi_csv),
            bbox_csv=str(bbox_csv),
            weather_csv=str(weather_csv),
            cloudless_mosaic_path=str(cloudless_mosaic_path),
            data_dir=str(data_dir_path),
            city_name=args.city_name,
            include_lst=args.include_lst,
            single_lst_median_path=str(single_lst_median_file) if single_lst_median_file else None
        )
    except FileNotFoundError as e:
        logging.error(f"Dataset initialization failed: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error during dataset initialization: {e}", exc_info=True)
        return

    # --- Train/Val Split ---
    val_percent = 0.15
    n_samples = len(dataset)
    if n_samples < 10: # Handle very small datasets
        logging.warning(f"Dataset size ({n_samples}) is very small. Consider disabling validation split or using more data.")
        n_val = 0
        n_train = n_samples
        train_ds = dataset
        val_ds = None # No validation set
    else:
        n_val = int(n_samples * val_percent)
        n_train = n_samples - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    logging.info(f"Dataset split: {n_train} training, {n_val or 0} validation samples.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None

    # --- Model Definition ---
    mosaic_channels = dataset.num_static_bands
    lst_channels = 1 if args.include_lst else 0

    try:
        model = UHINetConvGRU(
            mosaic_channels=mosaic_channels,
            weather_channels=3,
            time_emb_channels=4,
            lst_channels=lst_channels,
            proj_ch=args.proj_ch,
            hid_ch=args.hid_ch,
            freeze_clay=not args.unfreeze_clay
        ).to(device)
    except RuntimeError as e:
         logging.error(f"Failed to initialize model: {e}", exc_info=True)
         return
    except Exception as e:
        logging.error(f"Unexpected error initializing model: {e}", exc_info=True)
        return

    logging.info(f"Model initialized: {model.__class__.__name__}")
    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {num_params_total / 1e6:.2f} M")
    logging.info(f"Trainable parameters: {num_params_trainable / 1e6:.2f} M")

    # --- Loss and Optimizer ---
    loss_fn = masked_mae_loss if args.loss == 'mae' else masked_mse_loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = args.patience

    # Output directory
    run_name = f"{args.city_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoints and logs: {output_dir}")
    with open(output_dir / "args.json", 'w') as f: json.dump(vars(args), f, indent=2)

    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        if val_loader:
            val_loss = validate_epoch(model, val_loader, loss_fn, device)
            logging.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            current_loss = val_loss
        else:
             logging.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
             current_loss = train_loss # Use train loss for checkpointing if no val set

        is_best = current_loss < best_val_loss
        if is_best:
            best_val_loss = current_loss
            epochs_no_improve = 0
            logging.info(f"New best loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss,
             'optimizer' : optimizer.state_dict(), 'args': vars(args)},
            is_best,
            filename=output_dir / 'checkpoint_last.pth.tar',
            best_filename=output_dir / 'model_best.pth.tar'
        )

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    logging.info("Training finished.")
    logging.info(f"Best loss recorded: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UHI Prediction Model (Single Day UHI Version)")

    # Data Args
    parser.add_argument('city_name', type=str, help='Name of the city')
    parser.add_argument('--data_dir', type=str, default='data', help='Base data directory')
    parser.add_argument('--cloudless_mosaic_path', type=str, required=True, help='Path to the .npy cloudless mosaic file')
    parser.add_argument('--bounds', type=float, nargs=4, default=None, help='Optional: [min_lon, min_lat, max_lon, max_lat]. Tries bbox.csv if not given.')
    parser.add_argument('--resolution_m', type=int, default=10, help='Target resolution for grids (UHI, Weather, LST)')
    parser.add_argument('--include_lst', action='store_true', help='Include single LST median as input')
    parser.add_argument('--single_lst_median_path', type=str, default=None, help='Path to the pre-generated single LST median .npy file (Required if --include_lst)')
    # parser.add_argument('--averaging_window', type=int, default=30, help='Lookback window used to generate the single LST median (for info only)') # Removed, not used by dataloader

    # Model Args
    parser.add_argument('--proj_ch', type=int, default=64, help='Projection channels for Clay features')
    parser.add_argument('--hid_ch', type=int, default=64, help='Hidden channels for ConvGRU')
    parser.add_argument('--unfreeze_clay', action='store_true', help='Unfreeze Clay encoder weights for fine-tuning')

    # Training Args
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--loss', type=str, default='mae', choices=['mae', 'mse'], help='Loss function')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience') # Added patience arg
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='runs', help='Output directory for checkpoints/logs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()
    main(args) 