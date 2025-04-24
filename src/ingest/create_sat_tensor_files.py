import numpy as np
import os
import xarray as xr
import pandas as pd
import pystac_client
import planetary_computer
from tqdm import tqdm
from scipy.ndimage import zoom
from odc.stac import stac_load
from pathlib import Path
import logging
import json
from datetime import datetime
import s3fs
from botocore.exceptions import NoCredentialsError
import re # For potential filename parsing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def radiance_to_temperature_landsat(lwir_band):
    """Convert Landsat radiance to temperature."""
    K1_CONSTANT = 774.8853
    K2_CONSTANT = 1321.0789
    lwir_band = xr.where(lwir_band <= 0, np.nan, lwir_band)
    temperature = K2_CONSTANT / np.log((K1_CONSTANT / lwir_band) + 1)
    return temperature

def download_sentinel_data(city_name, bounds, time_windows, output_dir, 
                          selected_bands=["B02", "B03", "B04", "B08"], 
                          resolution_m=10):
    """
    Download Sentinel-2 data for specified city, time windows, and bounds.
    Saves median composite tensors to disk.
    
    Args:
        city_name: Name of the city (used for file naming)
        bounds: [min_lon, min_lat, max_lon, max_lat]
        time_windows: List of tuples (start_date, end_date) in format "YYYY-MM-DD"
        output_dir: Base directory to save data
        selected_bands: List of Sentinel-2 bands to download
        resolution_m: Spatial resolution in meters
    """
    # Create output directory
    city_dir = Path(output_dir) / city_name / "sat_files"
    city_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "city": city_name,
        "bounds": bounds,
        "bands": selected_bands,
        "resolution_m": resolution_m,
        "created_at": datetime.now().isoformat()
    }
    
    with open(city_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Process each time window
    for i, (start_date, end_date) in enumerate(tqdm(time_windows, desc=f"Downloading Sentinel for {city_name}")):
        time_window = f"{start_date}/{end_date}"
        output_filename = f"sentinel_{city_name}_{start_date}_{end_date}.npy"
        output_path = city_dir / output_filename
        
        # Skip if file already exists
        if output_path.exists():
            logging.info(f"File {output_path} already exists. Skipping.")
            continue
        
        logging.info(f"Processing Sentinel data for {city_name}: {time_window}")
        
        try:
            # Download and process sentinel data
            scale = resolution_m / 111320.0
            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = catalog.search(
                bbox=bounds,
                datetime=time_window,
                collections=["sentinel-2-l2a"],
                query={"eo:cloud_cover": {"lt": 30}}
            )
            items = list(search.items())
            
            if not items:
                logging.warning(f"No Sentinel data found for {city_name}: {time_window}")
                continue
                
            signed_items = [planetary_computer.sign(item) for item in items]
            ds = stac_load(
                signed_items,
                bands=selected_bands,
                crs="EPSG:4326",
                resolution=scale,
                chunks={"x": 2048, "y": 2048},
                patch_url=planetary_computer.sign,
                bbox=bounds
            )

            arr = ds.to_array()
            if "variable" in arr.dims:
                arr = arr.rename({"variable": "band"})
            if "latitude" in arr.dims or "longitude" in arr.dims:
                arr = arr.rename({"latitude": "y", "longitude": "x"})  

            arr = arr.transpose("band", "time", "y", "x")
            median_tensor = arr.median(dim="time", skipna=True)
            sentinel_data = median_tensor.transpose("band", "y", "x").values.astype(np.float32)
            
            # Save the data
            np.save(output_path, sentinel_data)
            logging.info(f"Saved Sentinel data to {output_path}")
            
            # Save timestamp info (consider removing if metadata.json covers this)
            # timestamp_info = {
            #     "start_date": start_date,
            #     "end_date": end_date,
            #     "shape": [int(s) for s in sentinel_data.shape], # Ensure serializable
            #     "file": output_filename
            # }
            # with open(city_dir / f"sentinel_{city_name}_{start_date}_{end_date}_info.json", "w") as f:
            #     json.dump(timestamp_info, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error processing Sentinel data for {city_name}: {time_window}")
            logging.error(str(e))

def download_lst_data(city_name, bounds, time_windows, output_dir, resolution_m=30):
    """
    Download Landsat Land Surface Temperature data for specified city and time windows.
    Saves median composite tensors to disk.
    
    Args:
        city_name: Name of the city (used for file naming)
        bounds: [min_lon, min_lat, max_lon, max_lat]
        time_windows: List of tuples (start_date, end_date) in format "YYYY-MM-DD"
        output_dir: Base directory to save data 
        resolution_m: Spatial resolution in meters
    """
    # Create output directory
    city_dir = Path(output_dir) / city_name / "sat_files"
    city_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each time window
    for i, (start_date, end_date) in enumerate(tqdm(time_windows, desc=f"Downloading Landsat LST for {city_name}")):
        time_window = f"{start_date}/{end_date}"
        output_filename = f"lst_{city_name}_{start_date}_{end_date}.npy"
        output_path = city_dir / output_filename
        
        # Skip if file already exists
        if output_path.exists():
            logging.info(f"File {output_path} already exists. Skipping.")
            continue
        
        logging.info(f"Processing LST data for {city_name}: {time_window}")
        
        try:
            # Download and process LST data
            scale = resolution_m / 111320.0
            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = catalog.search(
                bbox=bounds,
                datetime=time_window,
                collections=["landsat-c2-l2"],
                query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}}
            )
            items = list(search.items())
            
            if not items:
                logging.warning(f"No Landsat data found for {city_name}: {time_window}")
                continue
                
            signed_items = [planetary_computer.sign(item) for item in items]
            ds = stac_load(
                signed_items,
                bands=["lwir11"],
                crs="EPSG:4326",
                resolution=scale,
                bbox=bounds,
                chunks={"x": 2048, "y": 2048},
                patch_url=planetary_computer.sign
            )
            
            da = ds["lwir11"]
            lst_stack = []
            for j in range(da.sizes["time"]):
                radiance = da.isel(time=j)
                temperature = radiance_to_temperature_landsat(radiance)
                lst_stack.append(temperature.values)
                
            lst_stack = np.stack(lst_stack, axis=0)
            lst_median = np.nanmedian(lst_stack, axis=0, keepdims=True)
            lst_data = lst_median.astype(np.float32)
            
            # Save the data
            np.save(output_path, lst_data)
            logging.info(f"Saved LST data to {output_path}")
            
            # Save timestamp info (consider removing)
            # timestamp_info = {
            #     "start_date": start_date,
            #     "end_date": end_date,
            #     "shape": [int(s) for s in lst_data.shape],
            #     "file": output_filename
            # }
            # with open(city_dir / f"lst_{city_name}_{start_date}_{end_date}_info.json", "w") as f:
            #     json.dump(timestamp_info, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error processing LST data for {city_name}: {time_window}")
            logging.error(str(e))

def download_goes_hourly_data(city_name, hourly_timestamps, output_dir, goes_product="ABI-L2-LSTF"):
    """
    Downloads hourly GOES LST NetCDF files for given timestamps.

    Args:
        city_name (str): Name of the city (used for subdirectories).
        hourly_timestamps (list): List of pandas Timestamps for the required hours.
        output_dir (str or Path): Base directory to save the files.
        goes_product (str): GOES ABI L2+ product string (e.g., 'ABI-L2-LSTF').
    """
    try:
        s3 = s3fs.S3FileSystem(anon=True)
        goes_satellite = "noaa-goes16" # Or goes17/18 depending on location/time
        s3_base = f"{goes_satellite}/{goes_product}"
        logging.info(f"Attempting to download GOES data from S3 bucket: {s3_base}")
    except NoCredentialsError:
        logging.error("AWS credentials not found, but required by s3fs?")
        logging.error("Attempting anonymous access failed unexpectedly.")
        return
    except Exception as e:
        logging.error(f"Failed to initialize S3 filesystem: {e}")
        return

    # Create local output directory structure
    goes_output_dir = Path(output_dir) / city_name / "goes_files"
    goes_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"GOES data will be saved to: {goes_output_dir}")

    processed_hours = set()
    missing_files = 0
    downloaded_files = 0

    # Get unique timestamps to avoid redundant checks
    unique_timestamps = sorted(list(set(t.floor('h') for t in hourly_timestamps)))

    for ts in tqdm(unique_timestamps, desc=f"Downloading GOES {goes_product} for {city_name}"):
        year = ts.strftime('%Y')
        day_of_year = ts.strftime('%j')
        hour = ts.strftime('%H')

        local_filename = f"goes_{ts.strftime('%Y%m%d%H')}.nc"
        local_path = goes_output_dir / local_filename

        # Skip if already downloaded
        if local_path.exists():
            logging.debug(f"GOES file already exists: {local_path}. Skipping.")
            continue

        s3_hour_path = f"{s3_base}/{year}/{day_of_year}/{hour}/"

        try:
            # List files for that hour to find the exact filename
            # GOES filenames usually look like: OR_ABI-L2-LSTF-M6_G16_sYYYYJJJHHMMSSs_..._cYYYYJJJHHMMSSs.nc
            logging.debug(f"Checking S3 path: {s3_hour_path}")
            files_in_hour = s3.ls(s3_hour_path)

            # Find the file corresponding to the start of the hour (best guess)
            # Example pattern: look for _sYYYYJJJHH00
            hour_start_pattern = f"_s{year}{day_of_year}{hour}00"
            target_s3_file = None
            for f in files_in_hour:
                if hour_start_pattern in f:
                     # More specific check if needed (e.g., ensure M6 mode)
                     if f"_{goes_product}-M6_" in f and f.endswith(".nc"):
                         target_s3_file = f
                         break # Found likely file

            if target_s3_file:
                logging.info(f"Found target S3 file: {target_s3_file}")
                logging.info(f"Downloading to: {local_path}")
                s3.get(target_s3_file, str(local_path))
                downloaded_files += 1
            else:
                logging.warning(f"No suitable GOES file found in {s3_hour_path} for pattern {hour_start_pattern}")
                missing_files += 1

        except FileNotFoundError:
            logging.warning(f"S3 path not found or no files listed for: {s3_hour_path}")
            missing_files += 1
        except Exception as e:
            logging.error(f"Error downloading GOES for {ts.strftime('%Y-%m-%d %H')}: {e}")
            missing_files += 1

    logging.info(f"GOES download complete for {city_name}. Downloaded: {downloaded_files}, Missing/Errors: {missing_files}")

def download_satellite_data_for_city(city_name, bounds, timestamps,
                                     averaging_window, output_dir,
                                     selected_bands=["B02", "B03", "B04", "B08"],
                                     resolution_m=10, include_lst=True,
                                     include_goes=True): # Added GOES flag
    """
    Download Sentinel, LST (Landsat), and optionally GOES LST data.
    Args:
       # ... (existing args) ...
       include_goes (bool): Whether to download hourly GOES LST data.
    """
    # Convert timestamps to time windows
    time_windows = []
    for ts in timestamps:
        end_date = ts.strftime("%Y-%m-%d")
        start_date = (ts - pd.Timedelta(days=averaging_window)).strftime("%Y-%m-%d")
        time_windows.append((start_date, end_date))

    # Create unique time windows
    unique_time_windows = list(set(time_windows))

    # Convert timestamps list to hourly timestamps for GOES
    hourly_timestamps = list(set(t.floor('h') for t in timestamps))

    # Download Sentinel data
    download_sentinel_data(
        city_name=city_name,
        bounds=bounds,
        time_windows=unique_time_windows,
        output_dir=output_dir,
        selected_bands=selected_bands,
        resolution_m=resolution_m
    )

    # Download LST data if requested
    if include_lst:
        download_lst_data(
            city_name=city_name,
            bounds=bounds,
            time_windows=unique_time_windows,
            output_dir=output_dir,
            resolution_m=30  # LST typically at 30m resolution
        )

    # --- Download GOES data if requested ---
    if include_goes:
         # Use the same base output_dir, function will create city/goes_files subdir
         goes_output_dir = output_dir
         download_goes_hourly_data(
             city_name=city_name,
             hourly_timestamps=hourly_timestamps,
             output_dir=goes_output_dir
         )
    # --- End GOES download ---

    # Create lookup table for time windows to filenames
    lookup_table = {}
    city_dir = Path(output_dir) / city_name / "sat_files"

    for start_date, end_date in unique_time_windows:
        time_window = f"{start_date}/{end_date}"
        lookup_table[time_window] = {
            "sentinel": f"sentinel_{city_name}_{start_date}_{end_date}.npy",
            "lst": f"lst_{city_name}_{start_date}_{end_date}.npy" if include_lst else None
        }

    # Save lookup table
    with open(city_dir / "timewindow_lookup.json", "w") as f:
        json.dump(lookup_table, f, indent=2)

    return lookup_table

def download_data_from_uhi_csv(city_name, uhi_csv, bbox_csv,
                               averaging_window, output_dir,
                               selected_bands=["B02", "B03", "B04", "B08"],
                               resolution_m=10, include_lst=True,
                               include_goes=True): # Added GOES flag
    """
    Helper function to download satellite data based on UHI CSV file.
    Args:
       # ... (existing args) ...
       include_goes (bool): Whether to download hourly GOES LST data.
    """
    # Read UHI data and parse timestamps
    uhi_data = pd.read_csv(uhi_csv)
    timestamps = pd.to_datetime(uhi_data['timestamp'])

    # Read bounding box
    bbox_data = pd.read_csv(bbox_csv)
    min_lon = bbox_data['longitudes'].min()
    min_lat = bbox_data['latitudes'].min()
    max_lon = bbox_data['longitudes'].max()
    max_lat = bbox_data['latitudes'].max()
    bounds = [min_lon, min_lat, max_lon, max_lat]

    # Download the data
    return download_satellite_data_for_city(
        city_name=city_name,
        bounds=bounds,
        timestamps=timestamps,
        averaging_window=averaging_window,
        output_dir=output_dir,
        selected_bands=selected_bands,
        resolution_m=resolution_m,
        include_lst=include_lst,
        include_goes=include_goes # Pass the flag
    )

if __name__ == "__main__":
    # Example usage
    city_name = "NYC"
    # Construct absolute paths relative to the script's location if needed
    script_dir = Path(__file__).parent
    project_root_dir = script_dir.parent.parent # Adjust based on your structure

    uhi_csv = project_root_dir / f"data/{city_name}/uhi_data.csv"
    bbox_csv = project_root_dir / f"data/{city_name}/bbox.csv"
    averaging_window = 30
    output_dir = project_root_dir / "data"

    # Verify inputs exist before running
    if not uhi_csv.exists():
        logging.error(f"UHI CSV not found: {uhi_csv}")
    elif not bbox_csv.exists():
        logging.error(f"Bbox CSV not found: {bbox_csv}")
    else:
        logging.info("Starting data download process...")
        download_data_from_uhi_csv(
            city_name=city_name,
            uhi_csv=str(uhi_csv),
            bbox_csv=str(bbox_csv),
            averaging_window=averaging_window,
            output_dir=str(output_dir),
            include_lst=True, # Download Landsat LST
            include_goes=True  # Download GOES LST
        )
        logging.info("Data download process finished.") 