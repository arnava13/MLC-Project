import numpy as np
import os
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime
import argparse
from typing import Optional

# Assuming get_median contains the necessary loading function
from .get_median import load_lst_tensor_from_bbox_median # Relative import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_single_lst_median(city_name, bounds, output_dir,
                               uhi_csv_path: Optional[Path] = None,
                               averaging_window: Optional[int] = None,
                               time_window: Optional[str] = None, # Explicit time window
                               resolution_m=30, lst_cloud_cover=50):
    """
    Downloads a single Landsat LST median composite.

    Can determine the time window based on UHI CSV + lookback, OR use an
    explicitly provided time_window.

    Args:
        city_name (str): Name of the city.
        bounds (list): [min_lon, min_lat, max_lon, max_lat].
        output_dir (Path): Base directory to save data.
        uhi_csv_path (Path, optional): Path to UHI data CSV. Needed if time_window not given.
        averaging_window (int, optional): Lookback days. Needed if time_window not given.
        time_window (str, optional): Explicit time window ('YYYY-MM-DD/YYYY-MM-DD'). Overrides CSV lookup.
        resolution_m (int): Spatial resolution for LST download.
        lst_cloud_cover (int): Cloud cover threshold (Note: hardcoded in load_lst func).

    Returns:
        Path: Path to the generated LST median file, or None if failed.
    """
    city_dir = Path(output_dir) / city_name / "sat_files"
    city_dir.mkdir(parents=True, exist_ok=True)

    # Determine time window
    if time_window:
        logging.info(f"Using provided LST time window: {time_window}")
        start_date = time_window.split('/')[0]
        end_date = time_window.split('/')[1]
    elif uhi_csv_path and averaging_window is not None:
        logging.info("Determining LST time window from UHI CSV...")
        try:
            uhi_df = pd.read_csv(uhi_csv_path)
            timestamps = pd.to_datetime(uhi_df['timestamp'])
            if timestamps.empty:
                logging.error(f"No timestamps found in {uhi_csv_path}")
                return None
            latest_timestamp = timestamps.max()
            end_date = latest_timestamp.strftime("%Y-%m-%d")
            start_date = (latest_timestamp - pd.Timedelta(days=averaging_window)).strftime("%Y-%m-%d")
            time_window = f"{start_date}/{end_date}"
            logging.info(f"Determined single LST time window: {time_window}")
        except Exception as e:
            logging.error(f"Error reading UHI timestamps from {uhi_csv_path}: {e}")
            return None
    else:
        logging.error("Must provide either 'time_window' or ('uhi_csv_path' and 'averaging_window')")
        return None

    # Define output filename based on the final window
    start_date_str = start_date.replace('-', '')
    end_date_str = end_date.replace('-', '')
    output_filename = f"lst_{city_name}_median_{start_date_str}_to_{end_date_str}.npy"
    output_path = city_dir / output_filename

    if output_path.exists():
        logging.info(f"Single LST median file {output_path} already exists. Skipping generation.")
        return output_path

    logging.info(f"Attempting to generate single LST median for {city_name}: {time_window}")

    try:
        # Use the existing function from get_median to load the data
        lst_data = load_lst_tensor_from_bbox_median(
            bounds=bounds,
            time_window=time_window,
            resolution_m=resolution_m
            # Note: load_lst_tensor_from_bbox_median currently hardcodes cloud cover to 50
        )

        np.save(output_path, lst_data)
        logging.info(f"Saved single LST median to {output_path}")

        # Update metadata
        metadata_path = city_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f: metadata = json.load(f)
            except json.JSONDecodeError: pass # Ignore if corrupt

        metadata[f'single_lst_median_{start_date_str}_{end_date_str}'] = { # Use window in key
            "source_uhi_file": str(uhi_csv_path.name) if uhi_csv_path else None,
            "averaging_window_days": averaging_window if averaging_window is not None else None,
            "explicit_time_window": time_window,
            "resolution_m": resolution_m,
            "cloud_cover_threshold": 50,
            "file_generated": output_filename,
            "last_updated": datetime.now().isoformat()
        }
        if 'bounds' not in metadata: metadata['bounds'] = bounds
        with open(metadata_path, "w") as f: json.dump(metadata, f, indent=2)

        return output_path

    except ValueError as e:
        logging.error(f"No suitable Landsat data found for single LST median ({city_name}, {time_window}): {e}")
        return None
    except Exception as e:
        logging.error(f"Error generating single LST median for {city_name}, {time_window}: {e}")
        if output_path.exists(): output_path.unlink(missing_ok=True)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a single LST median composite.")
    parser.add_argument("city_name", type=str, help="Name of the city (e.g., NYC).")
    parser.add_argument("--bounds", required=True, type=float, nargs=4, metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'), help="Bounding box.")
    parser.add_argument("--output_dir", type=str, default="data", help="Base directory for data storage.")
    parser.add_argument("--lst_res", type=int, default=30, help="Resolution (meters) for LST download.")

    # Arguments for time window calculation
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--time_window", type=str, help="Explicit time window (YYYY-MM-DD/YYYY-MM-DD).")
    group.add_argument("--uhi_csv", type=str, help="Path to UHI CSV to derive time window (requires --averaging_window).")

    parser.add_argument("--averaging_window", type=int, help="Lookback days from latest UHI timestamp (only used with --uhi_csv).")

    args = parser.parse_args()

    # Validate arguments
    if args.uhi_csv and args.averaging_window is None:
        parser.error("--averaging_window is required when using --uhi_csv")

    uhi_csv_arg = Path(args.uhi_csv) if args.uhi_csv else None
    if uhi_csv_arg and not uhi_csv_arg.exists():
        logging.error(f"UHI CSV not found: {uhi_csv_arg}")
    else:
        logging.info("Starting single LST median download process...")
        download_single_lst_median(
            city_name=args.city_name,
            bounds=args.bounds,
            output_dir=Path(args.output_dir),
            uhi_csv_path=uhi_csv_arg,
            averaging_window=args.averaging_window,
            time_window=args.time_window,
            resolution_m=args.lst_res
        )
        logging.info("Single LST median download process finished.") 