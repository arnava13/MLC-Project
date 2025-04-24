import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import zoom
import json
import logging
import xarray as xr
import rasterio # Keep rasterio for potential GeoTIFF mosaic loading if needed

# Attempt to import MARSTFN, provide instructions if missing
try:
    import marstfn
except ImportError:
    logging.error("MARSTFN library not found. Please install it, e.g.:")
    logging.error("pip install git+https://github.com/yourorg/MARSTFN.git@main") # Use actual URL
    # Or handle the absence in a way suitable for your workflow (e.g., disable fusion)
    marstfn = None # Set to None or raise error if critical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CityDataSet(Dataset):
    """
    PyTorch Dataset for UHI modeling using satellite data.

    Loads Sentinel-2 median composites. If include_lst is True, it attempts
    to generate hourly high-resolution LST by fusing the Sentinel-2 composite
    with hourly GOES data using the MARSTFN model. Fused LST is cached.
    Falls back to zeros if fusion fails or inputs are missing.

    Returns (satellite_tensor, weather_tensor, meta_tensor) for each UHI observation.
    """

    def __init__(self, bounds, averaging_window, selected_bands, resolution_m,
                 include_lst=True, uhi_csv=None, bbox_csv=None, weather_csv=None,
                 data_dir=None, city_name=None,
                 goes_dir=None, # New: Directory for hourly GOES NetCDF files
                 fusion_dir=None, # New: Directory to cache fused LST outputs
                 marstfn_config_path=None): # New: Path to MARSTFN config JSON
        """
        Initialize the dataset with local satellite data files and fusion options.

        Args:
            bounds: Bounding box [min_lon, min_lat, max_lon, max_lat]
            averaging_window: Number of days to look back for median composites
            selected_bands: List of Sentinel-2 bands used
            resolution_m: Spatial resolution in meters for Sentinel data
            include_lst: Whether to include Land Surface Temperature data (via fusion)
            uhi_csv: Path to UHI data CSV file
            bbox_csv: Path to bounding box CSV file
            weather_csv: Path to weather data CSV file
            data_dir: Base directory for stored median satellite data (Sentinel, lookup)
            city_name: Name of the city (used for directory paths)
            goes_dir: Path to directory containing hourly GOES NetCDF files. Required if include_lst is True.
                      Expected naming: goes_YYYYMMDDHH.nc
            fusion_dir: Path to directory for caching MARSTFN fused LST outputs. Required if include_lst is True.
            marstfn_config_path: Path to the MARSTFN configuration JSON file. Required if include_lst is True.
        """
        # Set basic parameters
        self.bounds = bounds
        self.averaging_window = averaging_window
        self.selected_bands = selected_bands
        self.include_lst = include_lst
        self.resolution_m = resolution_m # Sentinel resolution
        self.bbox_csv = bbox_csv
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.city_name = city_name if city_name else "unknown_city"
        self.sat_files_dir = self.data_dir / self.city_name / "sat_files"

        # --- MARSTFN Fusion Parameters ---
        self.marstfn_config = None
        self.goes_dir = None
        self.fusion_dir = None
        if self.include_lst:
            if marstfn is None:
                 raise ImportError("MARSTFN library is required but not found/imported.")
            if not goes_dir:
                raise ValueError("`goes_dir` must be provided when `include_lst` is True.")
            if not fusion_dir:
                raise ValueError("`fusion_dir` must be provided when `include_lst` is True.")
            if not marstfn_config_path:
                 raise ValueError("`marstfn_config_path` must be provided when `include_lst` is True.")

            self.goes_dir = Path(goes_dir)
            self.fusion_dir = Path(fusion_dir)
            self.fusion_dir.mkdir(parents=True, exist_ok=True) # Ensure fusion cache dir exists

            # Load MARSTFN config
            try:
                with open(marstfn_config_path, 'r') as f:
                    self.marstfn_config = json.load(f)
                logging.info(f"Loaded MARSTFN config from {marstfn_config_path}")
            except Exception as e:
                logging.error(f"Failed to load MARSTFN config from {marstfn_config_path}: {e}")
                raise # Config is essential

            if not self.goes_dir.exists():
                 logging.warning(f"GOES directory not found: {self.goes_dir}")


        # Check if satellite data directory exists
        if not self.sat_files_dir.exists():
            raise ValueError(f"Satellite data directory not found: {self.sat_files_dir}")

        # Load lookup table for median Sentinel composites
        self.lookup_path = self.sat_files_dir / "timewindow_lookup.json"
        if not self.lookup_path.exists():
            raise ValueError(f"Timewindow lookup file not found: {self.lookup_path}")

        try:
            with open(self.lookup_path, 'r') as f:
                self.lookup_table = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load lookup table {self.lookup_path}: {e}")
            raise

        # Load UHI CSV and parse timestamp
        self.uhi_data = pd.read_csv(uhi_csv)
        # Ensure timestamp column exists and is parsed
        if 'timestamp' not in self.uhi_data.columns:
            raise ValueError("'timestamp' column missing in UHI CSV file.")
        try:
            # Store original timestamps for fusion reference
            self.hourly_timestamps = pd.to_datetime(self.uhi_data['timestamp'])
            self.uhi_data['timestamp_dt'] = self.hourly_timestamps # Keep datetime objects
        except Exception as e:
            logging.error(f"Failed to parse 'timestamp' column in {uhi_csv}: {e}")
            raise
        self.load_uhi_data() # Calculates grid indices, etc.

        # Load weather CSV (daily max/min temp + precipitation)
        self.weather_df = pd.read_csv(weather_csv)
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])

        # Preload Sentinel median tensors (LST will be loaded/fused on-the-fly or from cache)
        self.sentinel_tensors = self.load_sentinel_median_tensors()

    def load_uhi_data(self):
        """Load UHI data and compute grid coordinates."""
        # Compute grid index and time features from lat/lon and timestamp
        bbox_data = pd.read_csv(self.bbox_csv)
        # Assuming bbox_data has single row or consistent topleft for the city
        topleft_lat = bbox_data['latitudes'].iloc[0]
        topleft_lon = bbox_data['longitudes'].iloc[0]

        # Calculate resolution in degrees (approximation)
        deg_per_meter_lat = 1 / 111132.954 # Varies slightly with latitude
        deg_per_meter_lon = 1 / (111319.488 * np.cos(np.radians(topleft_lat))) # Varies with latitude

        # Use Sentinel resolution for grid calculation
        x_res_deg = self.resolution_m * deg_per_meter_lon
        y_res_deg = self.resolution_m * deg_per_meter_lat

        # Calculate grid indices relative to the top-left corner
        self.uhi_data['x_grid'] = np.floor((self.uhi_data['longitudes'] - topleft_lon) / x_res_deg).astype(int)
        self.uhi_data['y_grid'] = np.floor((topleft_lat - self.uhi_data['latitudes']) / y_res_deg).astype(int) # lat decreases downwards

        # Time features from the datetime objects
        self.uhi_data['min_since_midnight'] = self.uhi_data['timestamp_dt'].dt.hour * 60 + self.uhi_data['timestamp_dt'].dt.minute
        self.uhi_data['month'] = self.uhi_data['timestamp_dt'].dt.month

        # Keep necessary columns (including original timestamp_dt for fusion)
        self.uhi_data = self.uhi_data[['latitudes', 'longitudes', 'timestamp', 'timestamp_dt',
                                       'x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']]

    def get_time_window_str(self, timestamp):
        """Calculates the time window string based on the averaging window."""
        end_date = timestamp.strftime("%Y-%m-%d")
        start_date = (timestamp - pd.Timedelta(days=self.averaging_window)).strftime("%Y-%m-%d")
        return f"{start_date}/{end_date}"

    def apply_marstfn(self, timestamp):
        """
        Applies MARSTFN fusion for a specific hourly timestamp.

        Reads the corresponding Sentinel median mosaic and hourly GOES data,
        runs the fusion model, caches the output, and returns the fused LST tensor.

        Args:
            timestamp (pd.Timestamp): The specific hour for which to generate fused LST.

        Returns:
            np.ndarray: Fused LST tensor [1, H, W] at Sentinel resolution, or None if fusion fails.
        """
        if marstfn is None:
            logging.warning("MARSTFN library not available, cannot perform fusion.")
            return None

        time_window_str = self.get_time_window_str(timestamp)
        hourly_str = timestamp.strftime('%Y%m%d%H')
        fused_cache_filename = f"fused_{self.city_name}_{hourly_str}.npy"
        fused_cache_path = self.fusion_dir / fused_cache_filename

        # --- 1. Load Sentinel Median Composite ---
        sentinel_tensor = None
        try:
            if time_window_str in self.lookup_table:
                sentinel_filename = self.lookup_table[time_window_str]["sentinel"]
                sentinel_path = self.sat_files_dir / sentinel_filename
                if sentinel_path.exists():
                    sentinel_tensor = np.load(sentinel_path)
                    logging.debug(f"Loaded Sentinel median from {sentinel_path}")
                else:
                    logging.warning(f"Sentinel median file not found: {sentinel_path}")
            else:
                logging.warning(f"Time window {time_window_str} not in lookup table.")
        except Exception as e:
            logging.error(f"Error loading Sentinel median for {time_window_str}: {e}")

        if sentinel_tensor is None:
            return None # Cannot proceed without Sentinel data

        # --- 2. Load Hourly GOES Data ---
        goes_tensor = None
        goes_filename = f"goes_{hourly_str}.nc" # Assuming this naming convention
        goes_path = self.goes_dir / goes_filename
        try:
            if goes_path.exists():
                # Adapt this based on GOES NetCDF structure and MARSTFN input requirements
                with xr.open_dataset(goes_path) as ds:
                    # Example: Select LST variable, handle CRS/coords if needed
                    # goes_lst = ds['LST'].values # Adjust variable name
                    # Preprocessing might be needed (e.g., Kelvin to Celsius, cropping)
                    goes_tensor = ds # Pass the xarray dataset or required numpy array to MARSTFN
                logging.debug(f"Loaded GOES data from {goes_path}")
            else:
                logging.warning(f"Hourly GOES file not found: {goes_path}")
        except Exception as e:
            logging.error(f"Error loading GOES data for {hourly_str} from {goes_path}: {e}")

        if goes_tensor is None:
            return None # Cannot proceed without GOES data

        # --- 3. Run MARSTFN Fusion ---
        fused_lst = None
        try:
            logging.info(f"Running MARSTFN fusion for {timestamp}...")
            # This call is hypothetical - adapt based on the actual MARSTFN API
            fused_lst = marstfn.fuse(
                high_res_static=sentinel_tensor, # The median composite
                low_res_hourly=goes_tensor,      # The hourly GOES data
                config=self.marstfn_config       # Model configuration
            )
            # Expected output shape [1, H, W] matching Sentinel resolution

            if fused_lst is None or not isinstance(fused_lst, np.ndarray):
                 raise ValueError("MARSTFN fusion did not return a valid numpy array.")

            logging.info(f"MARSTFN fusion successful for {timestamp}. Output shape: {fused_lst.shape}")

            # --- 4. Cache Result ---
            try:
                np.save(fused_cache_path, fused_lst.astype(np.float32))
                logging.info(f"Cached fused LST to {fused_cache_path}")
            except Exception as e:
                logging.error(f"Failed to cache fused LST to {fused_cache_path}: {e}")
                # Continue, but fusion will re-run next time

            return fused_lst.astype(np.float32)

        except Exception as e:
            logging.error(f"MARSTFN fusion failed for timestamp {timestamp}: {e}")
            return None # Indicate failure


    def load_sentinel_median_tensors(self):
        """Loads only the Sentinel median tensors based on timestamps."""
        all_sentinel_tensors = []
        missing_data_count = 0

        for timestamp in self.hourly_timestamps: # Use the hourly timestamps
            time_window = self.get_time_window_str(timestamp)
            sentinel_tensor = None

            if time_window in self.lookup_table:
                try:
                    sentinel_filename = self.lookup_table[time_window]["sentinel"]
                    sentinel_path = self.sat_files_dir / sentinel_filename
                    if sentinel_path.exists():
                         sentinel_tensor = np.load(sentinel_path)
                    else:
                        logging.warning(f"Sentinel file not found: {sentinel_path} for window {time_window}")
                except Exception as e:
                    logging.error(f"Error loading Sentinel data for {time_window}: {e}")
            else:
                logging.warning(f"Time window {time_window} not found in lookup table for timestamp {timestamp}.")

            if sentinel_tensor is None:
                 # Append dummy tensor if Sentinel data is missing for the window
                 logging.warning(f"Using zeros for Sentinel data for window {time_window}.")
                 dummy_shape = (len(self.selected_bands), 1, 1) # Minimal dummy shape
                 sentinel_tensor = np.zeros(dummy_shape, dtype=np.float32)
                 missing_data_count += 1

            all_sentinel_tensors.append(sentinel_tensor)

        if missing_data_count > 0:
            logging.warning(f"Missing Sentinel median data for {missing_data_count} out of {len(self.hourly_timestamps)} time windows.")

        return all_sentinel_tensors


    def get_lst_tensor(self, timestamp):
        """Gets the LST tensor for a specific timestamp, using cache or fusion."""
        if not self.include_lst:
            return None

        hourly_str = timestamp.strftime('%Y%m%d%H')
        fused_cache_filename = f"fused_{self.city_name}_{hourly_str}.npy"
        fused_cache_path = self.fusion_dir / fused_cache_filename

        lst_tensor = None

        # 1. Check cache
        if fused_cache_path.exists():
            try:
                lst_tensor = np.load(fused_cache_path)
                logging.debug(f"Loaded fused LST from cache: {fused_cache_path}")
                # Basic shape check might be useful here
                if not isinstance(lst_tensor, np.ndarray) or lst_tensor.ndim < 3:
                    logging.warning(f"Invalid LST tensor loaded from cache {fused_cache_path}. Refusing.")
                    lst_tensor = None
                    fused_cache_path.unlink() # Remove invalid cache file
            except Exception as e:
                logging.warning(f"Error loading fused LST from cache {fused_cache_path}: {e}. Will attempt fusion.")
                lst_tensor = None # Ensure it's None if loading fails

        # 2. Apply MARSTFN if not cached or cache load failed
        if lst_tensor is None:
            logging.debug(f"Fused LST not in cache for {timestamp}. Attempting fusion.")
            lst_tensor = self.apply_marstfn(timestamp) # This handles caching on success

        # 3. Handle fusion failure
        if lst_tensor is None:
            logging.warning(f"Failed to obtain fused LST for timestamp {timestamp}. Using zeros.")
            # Determine expected shape based on a sample Sentinel tensor if possible
            # Placeholder: use minimal dummy shape [1, 1, 1]
            dummy_shape = (1, 1, 1)
            if self.sentinel_tensors and len(self.sentinel_tensors) > 0:
                 # Use shape of first available Sentinel tensor (H, W)
                 s_shape = self.sentinel_tensors[0].shape
                 if len(s_shape) == 3:
                     dummy_shape = (1, s_shape[1], s_shape[2])

            lst_tensor = np.zeros(dummy_shape, dtype=np.float32)

        return lst_tensor


    def get_weather_for(self, lat, lon, timestamp):
        """Retrieve weather info for the nearest grid point and date."""
        # Retrieve weather info (max/min temp, precip) for the nearest grid point and date
        date = pd.to_datetime(timestamp).normalize()
        tolerance = 0.005  # ~500m tolerance for lat/lon (adjustable)

        match = self.weather_df[
            (np.abs(self.weather_df['lat'] - lat) <= tolerance) &
            (np.abs(self.weather_df['lon'] - lon) <= tolerance) &
            (self.weather_df['date'] == date)
        ]

        if len(match) == 0:
            # If exact match fails, try finding nearest by date only? Or just return NaN
            logging.warning(f"No weather match for lat={lat}, lon={lon}, date={date.date()}")
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32) # Return NaNs

        # Return the first match found
        return match[['temp_max', 'temp_min', 'precip']].iloc[0].values.astype(np.float32)

    def __len__(self):
        """Number of UHI samples."""
        return len(self.uhi_data) # Base length on UHI samples

    def __getitem__(self, idx):
        """Return (satellite tensor, weather info, meta features) for sample at index."""
        uhi_row = self.uhi_data.iloc[idx]
        timestamp = uhi_row['timestamp_dt'] # Use the datetime object

        # Get preloaded Sentinel median tensor
        sentinel_tensor = self.sentinel_tensors[idx]

        # Get LST tensor (either fused, cached, or dummy)
        lst_tensor = self.get_lst_tensor(timestamp) # Handles None if not include_lst

        # Combine Sentinel and LST
        if lst_tensor is not None:
             # Resize LST to match Sentinel tensor shape if needed (MARSTFN *should* output correctly, but double check)
             if lst_tensor.shape[1:] != sentinel_tensor.shape[1:]:
                 zoom_factors = (
                     1,  # channel dimension
                     sentinel_tensor.shape[1] / lst_tensor.shape[1],  # H direction
                     sentinel_tensor.shape[2] / lst_tensor.shape[2],  # W direction
                 )
                 logging.debug(f"Resizing LST tensor from {lst_tensor.shape} to match Sentinel {sentinel_tensor.shape}")
                 try:
                     lst_tensor_resized = zoom(lst_tensor, zoom=zoom_factors, order=1) # Bilinear interpolation
                 except Exception as e:
                     logging.error(f"Failed to resize LST tensor for timestamp {timestamp}: {e}. Using original.")
                     lst_tensor_resized = lst_tensor # Fallback or handle differently

             else:
                 lst_tensor_resized = lst_tensor

             # Concatenate along the channel (band) dimension
             combined_satellite = np.concatenate([sentinel_tensor, lst_tensor_resized], axis=0)
        else:
            # Only Sentinel data if include_lst is False or fusion failed definitively
            combined_satellite = sentinel_tensor

        # Get weather data
        lat = uhi_row['latitudes']
        lon = uhi_row['longitudes']
        weather = self.get_weather_for(lat, lon, timestamp)

        # Get metadata
        meta = uhi_row[['x_grid', 'y_grid', 'min_since_midnight', 'month', 'UHI']].to_numpy(dtype=np.float32)

        # Convert to PyTorch tensors before returning (optional, depends on training loop)
        # combined_satellite_torch = torch.from_numpy(combined_satellite)
        # weather_torch = torch.from_numpy(weather)
        # meta_torch = torch.from_numpy(meta)
        # return combined_satellite_torch, weather_torch, meta_torch

        return combined_satellite, weather, meta # Return numpy arrays for now 