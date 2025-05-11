import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import logging
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.dataloader import CityDataSet
from src.model import UHINet
from src.train.loss import masked_mae_loss, masked_mse_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)
        logging.info(f"Saved new best model to {best_filename}")

# Training function
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)

    for batch in progress_bar:
        # Ensure all required keys are present
        required_keys = ['sentinel_mosaic', 'weather_seq', 'time_emb_seq', 'target', 'mask']
        if model.use_lst: # Check model's config
            required_keys.append('lst_seq')
        if not all(key in batch for key in required_keys):
            missing = [key for key in required_keys if key not in batch]
            logging.warning(f"Skipping batch due to missing keys: {missing}")
            continue

        # Move batch to device
        try:
            sentinel_mosaic = batch["sentinel_mosaic"].to(device)
            weather_seq = batch["weather_seq"].to(device)       # (B, T, C_weather, H, W)
            lst_seq = batch["lst_seq"].to(device) if model.use_lst else None # (B, T, C_lst, H, W) - T=1
            time_emb_seq = batch["time_emb_seq"].to(device)     # (B, T, C_time, H, W)
            target = batch["target"].to(device)               # (B, H, W)
            mask = batch["mask"].to(device)                   # (B, H, W)
        except Exception as e:
            logging.error(f"Error moving batch to device: {e}")
            continue # Skip batch if moving fails

        optimizer.zero_grad()
        try:
            # --- Refactored Training Logic --- 
            B, T, C_weather, H_in, W_in = weather_seq.shape
            _, _, C_time, _, _ = time_emb_seq.shape

            # 1. Encode and project static features ONCE
            static_lst_map = lst_seq[:, 0, :, :, :] if model.use_lst and lst_seq is not None else None # Get T=0 slice
            with torch.no_grad(): # Ensure Clay backbone remains frozen
                 static_features = model.encode_and_project_static(sentinel_mosaic, static_lst_map)
            _, C_static, H_feat, W_feat = static_features.shape
            
            # 2. Initialize hidden state
            h = torch.zeros(B, model.gru_hidden_dim, H_feat, W_feat, device=device)
            
            # 3. Resize dynamic features if needed
            if weather_seq.shape[3:] != (H_feat, W_feat):
                weather_seq_resized = F.interpolate(weather_seq.view(B*T, C_weather, H_in, W_in), size=(H_feat, W_feat), mode='bilinear', align_corners=False).view(B, T, C_weather, H_feat, W_feat)
            else:
                weather_seq_resized = weather_seq
            if time_emb_seq.shape[3:] != (H_feat, W_feat):
                time_emb_seq_resized = F.interpolate(time_emb_seq.view(B*T, C_time, H_in, W_in), size=(H_feat, W_feat), mode='bilinear', align_corners=False).view(B, T, C_time, H_feat, W_feat)
            else:
                time_emb_seq_resized = time_emb_seq
                
            # 4. Loop through time steps
            for t in range(T):
                weather_t = weather_seq_resized[:, t, :, :, :]      # (B, C_weather, H', W')
                time_emb_t = time_emb_seq_resized[:, t, :, :, :]    # (B, C_time, H', W')
                # Concatenate static + dynamic features
                x_t_combined = torch.cat([static_features, weather_t, time_emb_t], dim=1) 
                # Update hidden state
                h = model.step(x_t_combined, h)
            
            # 5. Predict from final hidden state
            prediction = model.predict(h) # (B, 1, H', W')

            # Resize prediction to target size if needed
            if prediction.shape[2:] != target.shape[1:]:
                 logging.warning(f"Prediction shape {prediction.shape} does not match target shape {target.shape}. Resizing prediction.")
                 prediction_resized = F.interpolate(prediction, size=target.shape[1:], mode='bicubic', align_corners=False)
            else:
                 prediction_resized = prediction
                 
            # Calculate loss
            loss = loss_fn(prediction_resized.squeeze(1), target, mask)

            # Check for NaN loss
            if torch.isnan(loss):
                 logging.warning("NaN loss detected, skipping backward pass.")
                 continue

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
            
        except RuntimeError as e:
             logging.error(f"Runtime error during training: {e}")
             if "out of memory" in str(e):
                 logging.error("CUDA out of memory. Try reducing batch size.")
             continue 
        except Exception as e:
             logging.error(f"Unexpected error during training step: {e}", exc_info=True)
             continue 

    return total_loss / num_batches if num_batches > 0 else 0.0

# Validation function
def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            # Ensure all required keys are present
            required_keys = ['sentinel_mosaic', 'weather_seq', 'time_emb_seq', 'target', 'mask']
            if model.use_lst: # Check model's config
                required_keys.append('lst_seq')
            if not all(key in batch for key in required_keys):
                missing = [key for key in required_keys if key not in batch]
                logging.warning(f"Skipping validation batch due to missing keys: {missing}")
                continue
            
            try:
                # Move batch to device
                sentinel_mosaic = batch["sentinel_mosaic"].to(device)
                weather_seq = batch["weather_seq"].to(device)       
                lst_seq = batch["lst_seq"].to(device) if model.use_lst else None
                time_emb_seq = batch["time_emb_seq"].to(device)     
                target = batch["target"].to(device)               
                mask = batch["mask"].to(device)   
                
                # --- Refactored Validation Logic --- 
                B, T, C_weather, H_in, W_in = weather_seq.shape
                _, _, C_time, _, _ = time_emb_seq.shape
    
                # 1. Encode static features ONCE
                static_lst_map = lst_seq[:, 0, :, :, :] if model.use_lst and lst_seq is not None else None 
                static_features = model.encode_and_project_static(sentinel_mosaic, static_lst_map)
                _, C_static, H_feat, W_feat = static_features.shape
                
                # 2. Initialize hidden state
                h = torch.zeros(B, model.gru_hidden_dim, H_feat, W_feat, device=device)
                
                # 3. Resize dynamic features if needed
                if weather_seq.shape[3:] != (H_feat, W_feat):
                    weather_seq_resized = F.interpolate(weather_seq.view(B*T, C_weather, H_in, W_in), size=(H_feat, W_feat), mode='bilinear', align_corners=False).view(B, T, C_weather, H_feat, W_feat)
                else:
                    weather_seq_resized = weather_seq
                if time_emb_seq.shape[3:] != (H_feat, W_feat):
                    time_emb_seq_resized = F.interpolate(time_emb_seq.view(B*T, C_time, H_in, W_in), size=(H_feat, W_feat), mode='bilinear', align_corners=False).view(B, T, C_time, H_feat, W_feat)
                else:
                    time_emb_seq_resized = time_emb_seq
                    
                # 4. Loop through time steps
                for t in range(T):
                    weather_t = weather_seq_resized[:, t, :, :, :]
                    time_emb_t = time_emb_seq_resized[:, t, :, :, :]
                    x_t_combined = torch.cat([static_features, weather_t, time_emb_t], dim=1)
                    h = model.step(x_t_combined, h)
                
                # 5. Predict from final hidden state
                prediction = model.predict(h)
                # --------------------------
                
                # Resize prediction to target size if needed
                if prediction.shape[2:] != target.shape[1:]:
                    prediction_resized = F.interpolate(prediction, size=target.shape[1:], mode='bilinear', align_corners=False)
                else:
                    prediction_resized = prediction

                # Calculate loss
                loss = loss_fn(prediction_resized.squeeze(1), target, mask)
                
                if torch.isnan(loss):
                    logging.warning("NaN validation loss detected, skipping batch.")
                    continue

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix(loss=loss.item())
                
            except Exception as e:
                 logging.error(f"Error during validation step: {e}", exc_info=True)
                 continue # Skip batch on error

    return total_loss / num_batches if num_batches > 0 else 0.0

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")

    # --- Output Directory Setup ---
    output_dir = project_root / 'training_runs'
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {output_dir}")

    # --- Path Setup & Checks ---
    data_dir_path = Path(args.data_dir)
    city_data_dir = data_dir_path / args.city_name
    uhi_csv = city_data_dir / "uhi.csv" # Standardized name
    bbox_csv = city_data_dir / "bbox.csv"
    # Station weather files
    bronx_weather_csv = city_data_dir / "bronx_weather.csv"
    manhattan_weather_csv = city_data_dir / "mahattan_weather.csv"
    cloudless_mosaic_path = Path(args.cloudless_mosaic_path)
    clay_checkpoint_path = Path(args.clay_checkpoint)
    clay_metadata_path = Path(args.clay_metadata)
    
    # LST Path (optional)
    single_lst_median_file = Path(args.single_lst_median_path) if args.include_lst and args.single_lst_median_path else None

    # Check essential files/dirs
    required_paths = {
        "UHI CSV": uhi_csv, "BBox CSV": bbox_csv,
        "Bronx Weather CSV": bronx_weather_csv, "Manhattan Weather CSV": manhattan_weather_csv,
        "Cloudless Mosaic": cloudless_mosaic_path, 
        "Clay Checkpoint": clay_checkpoint_path, "Clay Metadata": clay_metadata_path
    }
    if args.include_lst:
        required_paths["LST Median File"] = single_lst_median_file
        
    for name, path in required_paths.items():
        if path is None or not path.exists():
            logging.error(f"Required file/path missing for {name}: {path}")
            return
            
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
    logging.info("Initializing dataset...")
    try:
        dataset = CityDataSet(
            bounds=bounds,
            averaging_window=30, # Placeholder, not used with single LST path
            resolution_m=args.resolution_m,
            uhi_csv=str(uhi_csv),
            bbox_csv=str(bbox_csv),
            bronx_weather_csv=str(bronx_weather_csv),
            manhattan_weather_csv=str(manhattan_weather_csv),
            cloudless_mosaic_path=str(cloudless_mosaic_path),
            data_dir=str(data_dir_path),
            city_name=args.city_name,
            include_lst=args.include_lst,
            single_lst_median_path=str(single_lst_median_file) if single_lst_median_file else None
        )
    except Exception as e:
        logging.error(f"Error initializing dataset: {e}", exc_info=True)
        return

    # --- Train/Val Split ---
    val_percent = 0.15
    n_samples = len(dataset)
    if n_samples < 10:
        logging.warning(f"Dataset size ({n_samples}) is very small. Validation split disabled.")
        n_val = 0
        n_train = n_samples
        train_ds, val_ds = dataset, None
    else:
        n_val = int(n_samples * val_percent)
        n_train = n_samples - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    logging.info(f"Dataset split: {n_train} training, {n_val or 0} validation samples.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None

    # --- Initialize Model ---
    logging.info("Initializing model...")
    try:
        model = UHINet(
            # Clay args
            clay_checkpoint_path=str(clay_checkpoint_path),
            clay_metadata_path=str(clay_metadata_path),
            clay_model_size=args.clay_model_size,
            clay_bands=args.clay_bands,
            clay_platform=args.clay_platform,
            clay_gsd=args.clay_gsd,
            # Weather args
            weather_channels=5, # Hardcoded based on dataloader change
            # LST args
            lst_channels=1 if args.include_lst else 0,
            use_lst=args.include_lst,
            # Time embedding args
            time_embed_dim=2, # Hardcoded based on dataloader change
            # ConvGRU args
            proj_ch=args.proj_ch,
            gru_hidden_dim=args.gru_hidden_dim,
            gru_kernel_size=args.gru_kernel_size
        ).to(device)
    except Exception as e:
        logging.error(f"Error initializing model: {e}", exc_info=True)
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
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        if start_epoch > 0:
             logging.info(f"Resuming training from epoch {start_epoch}")
        else:
             logging.info("Starting training from scratch.")

    best_val_loss = float('inf')
    if args.resume:
         best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        logging.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # --- Checkpointing ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logging.info(f"New best validation loss: {best_val_loss:.4f}")

        # Define checkpoint paths within the output directory
        checkpoint_filename = output_dir / f'checkpoint_epoch_{epoch+1}.pth.tar'
        best_checkpoint_filename = output_dir / 'model_best.pth.tar'
        # Save latest checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args) # Save model config
            },
            is_best=is_best,
            filename=str(checkpoint_filename), # Pass as string
            best_filename=str(best_checkpoint_filename) # Pass as string
        )
        logging.info(f"Saved checkpoint to {checkpoint_filename}")
        # Optional: Prune old checkpoints here if desired

        # Add early stopping condition if needed
        # Example:
        # if patience_counter >= args.patience:
        #     logging.info("Early stopping triggered.")
        #     break

    logging.info("Training finished.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best model saved to: {best_checkpoint_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UHI Prediction Model")

    # Data Args
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory for city data")
    parser.add_argument("--city_name", type=str, required=True, help="Name of the city to train on")
    parser.add_argument("--cloudless_mosaic_path", type=str, required=True, help="Path to the pre-generated cloudless mosaic .npy file")
    parser.add_argument("--include_lst", action='store_true', help="Include LST data")
    parser.add_argument("--single_lst_median_path", type=str, help="Path to the single LST median .npy file (required if --include_lst)")
    parser.add_argument("--bounds", type=float, nargs=4, help="Bounding box [min_lon, min_lat, max_lon, max_lat] (optional, loads from bbox.csv if not provided)")
    parser.add_argument("--resolution_m", type=int, default=10, help="Target spatial resolution in meters")

    # Model Args
    parser.add_argument("--clay_checkpoint", type=str, required=True, help="Path to Clay model checkpoint (.ckpt)")
    parser.add_argument("--clay_metadata", type=str, required=True, help="Path to Clay metadata.yaml")
    parser.add_argument("--clay_model_size", type=str, default="large", help="Size of Clay model used")
    parser.add_argument("--clay_bands", nargs='+', default=["blue", "green", "red", "nir"], help="Band names for Clay input mosaic")
    parser.add_argument("--clay_platform", type=str, default="sentinel-2-l2a", help="Clay platform string")
    parser.add_argument("--clay_gsd", type=int, default=10, help="GSD of Clay input mosaic")
    parser.add_argument("--proj_ch", type=int, default=32, help="Channels after projecting Clay features")
    parser.add_argument("--gru_hidden_dim", type=int, default=64, help="Hidden dimension for ConvGRU cell")
    parser.add_argument("--gru_kernel_size", type=int, default=3, help="Kernel size for ConvGRU convolutions")
    # Removed --unfreeze_clay as model freezing is now hardcoded in UHINet

    # Training Args
    parser.add_argument("--output_dir", type=str, default="./training_runs", help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Optimizer weight decay")
    parser.add_argument("--loss", type=str, default="mae", choices=["mae", "mse"], help="Loss function type")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--cpu", action='store_true', help="Force CPU usage even if CUDA is available")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    main(args) 