import numpy as np
import pystac_client
import planetary_computer
from scipy.ndimage import zoom
import rasterio
import rasterio.warp
import rasterio.transform
import logging
import xarray as xr # Restore xarray for stac_load output processing
import rioxarray   # Add rioxarray for proper geospatial handling
from odc.stac import stac_load # Restore stac_load
import subprocess # For calling gdalwarp
import tempfile   # For temporary files
import os         # For path manipulation
from pathlib import Path # For path manipulation
import copy       # For deep copying items
import pystac     # Add pystac import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper to get asset href safely ---
def _get_asset_href(item_or_dict, band_name):
    """Safely gets the href for a given band asset from a STAC item object or dictionary."""
    assets = None
    item_id_str = "unknown_item"
    if isinstance(item_or_dict, dict):
        assets = item_or_dict.get('assets', {})
        item_id_str = item_or_dict.get("id", "unknown_item_dict")
    elif hasattr(item_or_dict, 'assets'): # Check if it's a pystac Item like object
        assets = item_or_dict.assets
        item_id_str = getattr(item_or_dict, 'id', "unknown_item_obj")
    else:
        logging.warning("Invalid type passed to _get_asset_href")
        return None

    asset = assets.get(band_name)
    href = None
    if asset:
        # Asset could be dict or Asset object
        href = asset.get('href') if isinstance(asset, dict) else getattr(asset, 'href', None)

    if href:
        return href

    # --- Fallbacks ---
    # Fallback for common band names (e.g., B04 vs B4)
    if band_name.startswith("B0") and len(band_name) == 3:
        alt_band_name = "B" + band_name[2]
        asset = assets.get(alt_band_name)
        if asset:
           href = asset.get('href') if isinstance(asset, dict) else getattr(asset, 'href', None)
           if href: return href
    # Fallback for lwir11 (this is now primary for LST, but keep general fallback logic)
    # if band_name == "lwir11":
    #      asset = assets.get("SR_B10") # Example if needed
    #      if asset:
    #         href = asset.get('href') if isinstance(asset, dict) else getattr(asset, 'href', None)
    #         if href: return href
    # Fallback for ST (No longer used for LST, keep for general robustness if needed)
    # if band_name == "ST":
    #      asset = assets.get("ST_B10")
    #      if asset:
    #          href = asset.get('href') if isinstance(asset, dict) else getattr(asset, 'href', None)
    #          if href: return href

    logging.warning(f"Asset for band '{band_name}' not found or has no href in item {item_id_str}")
    return None

def radiance_to_temperature_landsat(lwir_band_data):
    """Calculates temperature from Landsat lwir11 band data.
       Applies the specific scale/offset for Landsat C2 L2 products.
       Accepts xarray DataArray as input.
    """
    # Ensure input is xarray DataArray
    if not isinstance(lwir_band_data, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")

    # --- Apply known scale/offset for landsat-c2-l2 lwir11 band --- 
    # These values convert the scaled DNs to Kelvin temperature.
    scale = 0.00341802
    offset = 149.0
    logging.info(f"Applying fixed scale ({scale}) and offset ({offset}) for lwir11 LST conversion.")

    # Get nodata value if present
    nodata_val = lwir_band_data.attrs.get('nodata', None)
    # Create NaN mask based on nodata value before scaling
    nan_mask = None
    if nodata_val is not None:
        # Ensure comparison is done with correct dtype if necessary
        if lwir_band_data.dtype != np.dtype(type(nodata_val)):
            try:
                nodata_val = np.dtype(lwir_band_data.dtype).type(nodata_val)
            except Exception as e:
                logging.warning(f"Could not cast nodata value {nodata_val} to dtype {lwir_band_data.dtype}: {e}")
                nodata_val = None # Disable nodata masking if cast fails

        if nodata_val is not None:
             nan_mask = (lwir_band_data == nodata_val)

    # Apply scale and offset to get Temperature in Kelvin
    # Ensure data is float before scaling to avoid integer overflow/truncation
    temperature = (lwir_band_data.astype(float) * scale) + offset
    # -----------------------------------------------------------------

    # Apply NaN mask after calculations
    if nan_mask is not None:
        temperature = xr.where(nan_mask, np.nan, temperature)

    return temperature # Usually in Kelvin


def load_sentinel_tensor_from_bbox_median(bounds, time_window, selected_bands=["B02", "B03", "B04", "B08"], resolution_m=10, cloud_cover=30, dest_crs="EPSG:4326"):
    """
    Loads Sentinel-2 L2A data using odc.stac.stac_load, pre-processing GCP-referenced
    sources with gdalwarp into temporary files.
    """
    processed_items = [] # Now stores pystac.Item objects
    gdal_env = os.environ.copy()
    gdal_env['GDAL_HTTP_REDIRECTION'] = 'YES'
    gdal_env['GDAL_HTTP_USERAGENT'] = 'rasterio/odc-stac' # Identify client

    try:
        # Approximate degrees per meter for target resolution calculation
        deg_per_meter_lat = 1 / 111000
        center_lat = (bounds[1] + bounds[3]) / 2.0
        deg_per_meter_lon = 1 / (111320 * np.cos(np.radians(center_lat)))
        scale_lon = resolution_m * deg_per_meter_lon
        scale_lat = resolution_m * deg_per_meter_lat
        target_resolution = scale_lat # Use latitude scale for stac_load resolution parameter

        # Use modifier for signing, get pystac.Item objects
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            bbox=bounds,
            datetime=time_window,
            collections=["sentinel-2-l2a"],
            query={"eo:cloud_cover": {"lt": cloud_cover}}
        )
        items = list(search.items()) # Get pystac.Item objects
        if not items:
            logging.warning(f"No Sentinel data found for window {time_window} and bounds {bounds}")
            return None
        logging.debug(f"Found {len(items)} Sentinel-2 items. Checking for GCPs...")

        with tempfile.TemporaryDirectory(prefix="gdal_preprocess_") as tmpdir:
            logging.info(f"Using temporary directory for gdalwarp: {tmpdir}")
            tmpdir_path = Path(tmpdir)

            for i, item_obj in enumerate(items):
                item_id = item_obj.id
                item_needs_copy = False # Flag to track if we need to copy this item
                item_copy = None # Placeholder for the copied item

                for band in selected_bands:
                    # Access asset href directly from item object
                    asset = item_obj.assets.get(band)
                    alt_band_name = None # Track alternative name if used
                    if not asset or not asset.href:
                        # Check common alternative band names (e.g., B4 instead of B04)
                        if band.startswith("B0") and len(band) == 3:
                            alt_band_name = "B" + band[2]
                            asset = item_obj.assets.get(alt_band_name)
                            if not asset or not asset.href:
                                logging.warning(f"Asset '{band}' or '{alt_band_name}' missing/no href in item {item_id}. Skipping band check.")
                                continue
                        else:
                             logging.warning(f"Asset '{band}' missing or has no href in item {item_id}. Skipping band check.")
                             continue

                    original_href = asset.href

                    try:
                        # Check if GCPs are used
                        with rasterio.open(f'/vsicurl/{original_href}') as src:
                            if src.gcps[0] and src.transform.is_identity:
                                logging.debug(f"Item {item_id} asset '{band}' uses GCPs. Pre-processing with gdalwarp.")

                                # Create a deep copy of the item if we haven't already for this item
                                if not item_needs_copy:
                                    item_copy = copy.deepcopy(item_obj)
                                    item_needs_copy = True

                                # Define temporary output path
                                temp_output_filename = f"{item_id}_{band}_warped.tif"
                                temp_output_path = tmpdir_path / temp_output_filename

                                # Construct gdalwarp command
                                cmd = [
                                    'gdalwarp',
                                    '-r', 'bilinear',
                                    '-of', 'GTiff',
                                    '-t_srs', dest_crs,
                                    f'/vsicurl/{original_href}',
                                    str(temp_output_path)
                                ]
                                logging.debug(f"Running command: {' '.join(cmd)}")

                                # Run gdalwarp
                                result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=gdal_env)

                                if result.returncode == 0:
                                    logging.info(f"Successfully warped {item_id} asset '{band}' to {temp_output_path}")

                                    # --- Verification Step --- 
                                    try:
                                        with rasterio.open(temp_output_path) as warped_src:
                                            if warped_src.transform.is_identity:
                                                logging.warning(f"*** WARNING: gdalwarp output {temp_output_path} still has identity transform! Transform: {warped_src.transform}")
                                            else:
                                                logging.debug(f"Verified non-identity transform for {temp_output_path}: {warped_src.transform}")
                                    except Exception as verify_e:
                                        logging.error(f"Error verifying gdalwarp output {temp_output_path}: {verify_e}")
                                    # --- End Verification --- 

                                    # Modify the href in the item COPY's asset dictionary
                                    asset_key_to_modify = alt_band_name if alt_band_name and alt_band_name in item_copy.assets else band
                                    if asset_key_to_modify in item_copy.assets:
                                        item_copy.assets[asset_key_to_modify].href = temp_output_path.as_uri()
                                    else:
                                         logging.warning(f"Could not find asset key '{asset_key_to_modify}' in copied item {item_id} to update href.")
                                else:
                                    logging.error(f"gdalwarp failed for {item_id} asset '{band}'. Error: {result.stderr}")
                                    # If warp fails, we keep the original item (or copy with original hrefs)

                    except rasterio.RasterioIOError as e:
                        logging.error(f"Rasterio error checking {item_id} asset '{band}' at {original_href}: {e}.")
                    except Exception as e:
                        logging.error(f"Unexpected error checking {item_id} asset '{band}': {e}.")

                # Append the original item or the modified copy
                if item_needs_copy:
                    processed_items.append(item_copy)
                else:
                    processed_items.append(item_obj)

            # Call stac_load with the processed pystac.Item objects
            if not processed_items:
                logging.error("No items were processed successfully.")
                return None

            logging.debug(f"Loading data with odc.stac.stac_load on {len(processed_items)} items...")
            # --- Detailed pre-load logging and additional georeference checking ---
            items_with_issues = []
            for item_idx, p_item in enumerate(processed_items):
                has_georef_issues = False
                logging.debug(f"  Pre-load check for item {item_idx + 1}/{len(processed_items)}: ID {p_item.id}")
                for band_name in selected_bands:
                    asset_to_check = p_item.assets.get(band_name)
                    if asset_to_check and asset_to_check.href:
                        href_to_check = asset_to_check.href
                        # Determine if it's a local file (from gdalwarp) or remote URL
                        is_local_file = href_to_check.startswith('file://')
                        path_for_rasterio = href_to_check[7:] if is_local_file else f'/vsicurl/{href_to_check}'
                        try:
                            with rasterio.open(path_for_rasterio) as src_check:
                                has_gcps = bool(src_check.gcps[0]) if src_check.gcps else False
                                is_identity = src_check.transform.is_identity
                                
                                # Check for problematic georeferencing
                                if is_identity and not has_gcps:
                                    has_georef_issues = True
                                    logging.warning(
                                        f"    ⚠️ Item {p_item.id}, Band {band_name}: MISSING GEOREFERENCING - "
                                        f"identity transform without GCPs"
                                    )
                                
                                logging.debug(
                                    f"    Item {p_item.id}, Band {band_name}: href={href_to_check}, "
                                    f"transform={src_check.transform}, crs={src_check.crs}, "
                                    f"gcps={has_gcps}, "
                                    f"is_identity={is_identity}"
                                )
                        except Exception as e_check:
                            logging.error(
                                f"    Item {p_item.id}, Band {band_name}: ERROR opening asset {href_to_check} for pre-load check: {e_check}"
                            )
                    else:
                        logging.warning(f"    Item {p_item.id}, Band {band_name}: Asset missing or no href.")
                
                # Collect items with issues for potential removal
                if has_georef_issues:
                    items_with_issues.append((item_idx, p_item))
            
            # Optionally remove problematic items (uncomment if needed)
            if items_with_issues:
                logging.warning(f"Found {len(items_with_issues)} items with georeferencing issues.")
                # If all items have issues, we can't remove them all
                if len(items_with_issues) < len(processed_items):
                    logging.warning("Removing items with georeferencing issues to avoid reprojection warnings.")
                    # Remove items with issues, in reverse order to maintain valid indices
                    for item_idx, _ in sorted(items_with_issues, key=lambda x: x[0], reverse=True):
                        processed_items.pop(item_idx)
                    logging.warning(f"After removal: {len(processed_items)} items remain for processing.")
                else:
                    logging.warning("All items have georeferencing issues! Proceeding with caution.")
            # --- End detailed pre-load logging and checking ---

            # patch_url is likely not needed as hrefs are either signed URLs or file URIs
            ds = stac_load(
                processed_items, # List of pystac.Item objects
                bands=selected_bands,
                crs=dest_crs,
                resolution=target_resolution,
                chunks={"x": 2048, "y": 2048},
                bbox=bounds,
                fail_on_error=False
            )

        # Check if dataset is empty
        if not ds.data_vars:
            logging.error("stac_load returned an empty dataset. Check previous errors.")
            return None

        # Process dataset (median, convert to numpy)
        bands_in_ds = [b for b in selected_bands if b in ds.data_vars]
        if not bands_in_ds:
            logging.error(f"None of the selected bands {selected_bands} are in the loaded dataset.")
            return None
        if len(bands_in_ds) < len(selected_bands):
            logging.warning(f"Missing some bands in loaded dataset. Available: {bands_in_ds}")

        arr = ds[bands_in_ds].to_array(dim="band")

        if "latitude" in arr.dims and "longitude" in arr.dims:
            arr = arr.rename({"latitude": "y", "longitude": "x"})
        elif "lat" in arr.dims and "lon" in arr.dims:
            arr = arr.rename({"lat": "y", "lon": "x"})

        try:
            arr = arr.transpose("band", "time", "y", "x")
        except ValueError as e:
            logging.error(f"Failed to transpose dimensions. Current dims: {arr.dims}. Error: {e}")
            if "time" not in arr.dims:
                arr = arr.transpose("band", "y", "x")
                arr = arr.expand_dims("time", axis=1)
            else: raise e
            
        # Ensure we have rioxarray for proper geospatial handling
        import rioxarray  # Make sure this is imported
        
        # Get dimensions for proper transform creation
        width = arr.sizes.get('x')
        height = arr.sizes.get('y')
        
        if width and height and bounds:
            # Create the transform properly using the bounds like in example notebooks
            gt = rasterio.transform.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], width, height)
            
            # Convert to a DataArray with spatial dimensions for proper handling
            # This is a key step - using rioxarray to handle the geospatial metadata
            xarr = xr.DataArray(arr)
            
            # Apply the geospatial information directly as in example notebooks
            xarr = xarr.rio.write_crs("EPSG:4326", inplace=False)
            xarr = xarr.rio.write_transform(transform=gt, inplace=False)
            
            # Replace the original array with the properly georeferenced one
            arr = xarr
            
            logging.info(f"Applied proper transform using rioxarray: {gt}")
        else:
            logging.warning("Could not create transform from bounds - missing dimensions or bounds")

        # Calculate median with skipna=True to handle NaN values in the input data
        median_tensor = arr.median(dim="time", skipna=True)
        
        # Get the values and convert to float32 for storage efficiency
        final_median_tensor = median_tensor.values.astype(np.float32)
        
        # Explicitly set nodata values to NaN
        # First, check if there's a _FillValue or nodata attribute in arr
        nodata_value = None
        for var_name in arr.data_vars if hasattr(arr, 'data_vars') else []:
            if hasattr(arr[var_name], 'attrs') and ('_FillValue' in arr[var_name].attrs or 'nodata' in arr[var_name].attrs):
                nodata_value = arr[var_name].attrs.get('_FillValue', arr[var_name].attrs.get('nodata'))
                break
        
        # If a nodata value was found, replace it with NaN in the final tensor
        if nodata_value is not None:
            final_median_tensor = np.where(
                final_median_tensor == nodata_value, 
                np.nan, 
                final_median_tensor
            )
        
        # Check for any invalid values (inf, -inf) and set them to NaN as well
        final_median_tensor = np.where(
            np.isfinite(final_median_tensor),
            final_median_tensor,
            np.nan
        )
        
        logging.info(f"Generated median tensor with shape: {final_median_tensor.shape} with NaN for missing values")
        
        # Return the tensor (now with NaN for missing values)
        return final_median_tensor

    except FileNotFoundError:
        logging.error("gdalwarp command not found. Please ensure GDAL is installed and in the system PATH.")
        return None
    except Exception as e:
        logging.error(f"An error occurred during Sentinel loading: {e}", exc_info=True)
        return None


def load_lst_tensor_from_bbox_median(bounds, time_window, resolution_m=30, dest_crs="EPSG:4326"):
    """
    Loads Landsat L2 LST data using odc.stac.stac_load, pre-processing GCP-referenced
    sources with gdalwarp into temporary files.
    """
    processed_items = [] # Stores pystac.Item objects
    gdal_env = os.environ.copy()
    gdal_env['GDAL_HTTP_REDIRECTION'] = 'YES'
    gdal_env['GDAL_HTTP_USERAGENT'] = 'rasterio/odc-stac'

    try:
        # Approximate target resolution in degrees
        deg_per_meter_lat = 1 / 111000
        center_lat = (bounds[1] + bounds[3]) / 2.0
        deg_per_meter_lon = 1 / (111320 * np.cos(np.radians(center_lat)))
        # scale_lon = resolution_m * deg_per_meter_lon # Not directly used by stac_load resolution
        scale_lat = resolution_m * deg_per_meter_lat
        target_resolution = scale_lat

        # Use lwir11 based on example notebook
        lst_band_asset_key = "lwir11"

        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            bbox=bounds,
            datetime=time_window,
            # Use landsat-c2-l2 collection as specified in example
            collections=["landsat-c2-l2"],
            query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8", "landsat-9"]}}
        )
        items = list(search.items()) # Get pystac.Item objects
        if not items:
            logging.warning(f"No Landsat L2 data found for window {time_window} and bounds {bounds}")
            return None
        logging.info(f"Found {len(items)} Landsat L2 items. Checking LST band ('{lst_band_asset_key}') for GCPs...")

        with tempfile.TemporaryDirectory(prefix="gdal_lst_preprocess_") as tmpdir:
            logging.info(f"Using temporary directory for LST gdalwarp: {tmpdir}")
            tmpdir_path = Path(tmpdir)

            for i, item_obj in enumerate(items):
                item_id = item_obj.id
                item_needs_copy = False
                item_copy = None

                # Check for the specific LST band asset
                asset_to_process = item_obj.assets.get(lst_band_asset_key)
                if not asset_to_process or not asset_to_process.href:
                    logging.warning(f"Asset '{lst_band_asset_key}' missing/no href in item {item_id}. Skipping item.")
                    continue # Skip item if LST band is missing

                original_href = asset_to_process.href

                try:
                    # Check if GCPs are used
                    with rasterio.open(f'/vsicurl/{original_href}') as src:
                        if src.gcps[0] and src.transform.is_identity:
                            logging.info(f"Item {item_id} asset '{lst_band_asset_key}' uses GCPs. Pre-processing with gdalwarp.")
                            if not item_needs_copy:
                                item_copy = copy.deepcopy(item_obj)
                                item_needs_copy = True

                            temp_output_filename = f"{item_id}_{lst_band_asset_key}_warped.tif"
                            temp_output_path = tmpdir_path / temp_output_filename
                            cmd = [
                                'gdalwarp',
                                '-r', 'bilinear',
                                '-of', 'GTiff',
                                '-t_srs', dest_crs,
                                f'/vsicurl/{original_href}',
                                str(temp_output_path)
                            ]
                            logging.debug(f"Running command: {' '.join(cmd)}")
                            result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=gdal_env)

                            if result.returncode == 0:
                                logging.info(f"Successfully warped {item_id} asset '{lst_band_asset_key}' to {temp_output_path}")
                                # Modify href in the item COPY's assets dictionary
                                if lst_band_asset_key in item_copy.assets:
                                     item_copy.assets[lst_band_asset_key].href = temp_output_path.as_uri()
                                else:
                                     logging.warning(f"Could not find asset key '{lst_band_asset_key}' in copied item {item_id} to update href.")
                            else:
                                logging.error(f"gdalwarp failed for {item_id} asset '{lst_band_asset_key}'. Error: {result.stderr}")
                                # Decide how to handle warp failure - skip item?
                                item_needs_copy = False # Don't add the failed copy
                                continue # Skip to next item if warp failed

                except rasterio.RasterioIOError as e:
                    logging.error(f"Rasterio error checking {item_id} asset '{lst_band_asset_key}' at {original_href}: {e}. Skipping item.")
                    continue # Skip item if check fails
                except Exception as e:
                    logging.error(f"Unexpected error checking {item_id} asset '{lst_band_asset_key}': {e}. Skipping item.")
                    continue # Skip item if check fails

                # Append the original item or the modified copy (only if warp succeeded or wasn't needed)
                if item_needs_copy:
                    processed_items.append(item_copy)
                else:
                    processed_items.append(item_obj)

            if not processed_items:
                 logging.error("No LST items could be processed successfully.")
                 return None

            # Call stac_load with processed pystac.Item objects
            logging.info(f"Loading LST data (band: {lst_band_asset_key}) with odc.stac.stac_load...")

            ds = stac_load(
                processed_items,
                bands=[lst_band_asset_key],
                crs=dest_crs,
                resolution=target_resolution,
                chunks={"x": 2048, "y": 2048},
                bbox=bounds,
                fail_on_error=False
            )

        # --- Process LST Data ---
        if not ds.data_vars or lst_band_asset_key not in ds.data_vars:
             logging.error(f"LST band ('{lst_band_asset_key}') not found in loaded dataset variables: {list(ds.data_vars)}. Check stac_load errors.")
             return None

        lst_da = ds[lst_band_asset_key]

        # --- Calculate Median LST in Kelvin ---
        temperature_k = radiance_to_temperature_landsat(lst_da)
        median_lst_k = temperature_k.median(dim="time", skipna=True)
        final_median_lst = median_lst_k.values.astype(np.float32)
        
        # Get nodata value if present
        nodata_value = None
        if hasattr(lst_da, 'attrs') and ('_FillValue' in lst_da.attrs or 'nodata' in lst_da.attrs):
            nodata_value = lst_da.attrs.get('_FillValue', lst_da.attrs.get('nodata'))
        
        # Set nodata values to NaN
        if nodata_value is not None:
            final_median_lst = np.where(
                final_median_lst == nodata_value,
                np.nan,
                final_median_lst
            )
        
        # Check for any invalid values (inf, -inf, unreasonable values) and set them to NaN
        # For LST in Kelvin, reasonable range is ~200K to ~350K (approx -73°C to 77°C)
        final_median_lst = np.where(
            np.logical_and(
                np.isfinite(final_median_lst),
                np.logical_and(final_median_lst > 200, final_median_lst < 350)
            ),
            final_median_lst,
            np.nan
        )
        
        # Add channel dimension
        final_median_lst = final_median_lst[np.newaxis, :, :] # Shape (1, height, width)
        logging.info(f"Generated median LST tensor (Kelvin) with shape: {final_median_lst.shape} with NaN for missing values")
        return final_median_lst

    except FileNotFoundError:
        logging.error("gdalwarp command not found. Please ensure GDAL is installed and in the system PATH.")
        return None
    except Exception as e:
        logging.error(f"An error occurred during LST loading: {e}", exc_info=True)
        return None


def create_and_save_cloudless_mosaic(city_name, bounds, output_dir,
                                       time_window: str,
                                       selected_bands=["B02", "B03", "B04", "B08"],
                                       resolution_m=10, cloud_cover=5):
    """Generates and saves a cloudless Sentinel-2 median mosaic for a given time window."""
    from pathlib import Path

    sat_files_dir = Path(output_dir) / city_name / "sat_files"
    sat_files_dir.mkdir(parents=True, exist_ok=True)

    start_date_str = time_window.split('/')[0].replace('-', '')
    end_date_str = time_window.split('/')[1].replace('-', '')
    output_filename = f"sentinel_{city_name}_{start_date_str}_to_{end_date_str}_cloudless_mosaic.npy"
    output_path = sat_files_dir / output_filename

    if output_path.exists():
        logging.info(f"Cloudless mosaic {output_path} already exists. Skipping generation.")
        return output_path

    logging.info(f"Generating cloudless mosaic for {city_name}, window {time_window}...")
    try:
        # Call the function to get mosaic data
        mosaic_data = load_sentinel_tensor_from_bbox_median(
            bounds=bounds,
            time_window=time_window,
            selected_bands=selected_bands,
            resolution_m=resolution_m,
            cloud_cover=cloud_cover
        )
        if mosaic_data is None:
             raise ValueError("Mosaic generation returned None (likely processing error).")

        # Calculate median over time dimension
        if "time" in mosaic_data.dims:
            median_bands_tensor = mosaic_data.median("time", skipna=True) # NaNs are preserved if all inputs are NaN for a pixel
        else:
            median_bands_tensor = mosaic_data # No time dimension, use as is
            logging.warning("No 'time' dimension found in Sentinel array for median calculation.")
        
        # Ensure the data is float32. NaNs will be preserved.
        median_bands_tensor_float = median_bands_tensor.astype(np.float32)
        
        # Use the tensor that preserves NaNs for saving.
        output_numpy_array = median_bands_tensor_float.data 

        # Save the mosaic tensor data
        np.save(output_path, output_numpy_array)
        logging.info(f"Saved cloudless mosaic to {output_path}")
        
        # Save metadata in a JSON file if needed
        # Currently not saving metadata because we know the dataloader is creating its own transform
        # but we could uncomment this code if we want to include metadata in the future
        # metadata_path = output_path.with_suffix('.json')
        # transform_json = None
        # if transform is not None:
        #     transform_json = transform.to_gdal()
        # metadata = {
        #     'bounds': bounds,
        #     'transform': transform_json,
        #     'crs': str(crs) if crs else 'EPSG:4326',
        #     'shape': mosaic_data.shape
        # }
        # with open(metadata_path, 'w') as f:
        #     json.dump(metadata, f)
        # logging.info(f"Saved metadata to {metadata_path}")
        
        return output_path
    except ValueError as e:
        logging.error(f"No suitable Sentinel data found or processed for mosaic ({city_name}, {time_window}): {e}")
        return None
    except Exception as e:
        logging.error(f"Error generating cloudless mosaic for {city_name}, {time_window}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

######################################################
# Test the functions
# Uncomment the following lines to test the functions
######################################################
# bounds = [-74.01, 40.75, -73.86, 40.88]  # NYC
# time_window = "2021-06-01/2021-09-01"
# resolution = 10  

# # Sentinel-2 median
# sentinel = load_sentinel_tensor_from_bbox_median(bounds, time_window)
# print("Sentinel median tensor shape:", sentinel.shape)

# # LST median
# lst = load_lst_tensor_from_bbox_median(bounds, time_window)
# print("LST median tensor shape:", lst.shape)

# # combined tensor
# zoom_factors = (
#     1,  # channel dimension
#     sentinel.shape[1] / lst.shape[1],  # H direction
#     sentinel.shape[2] / lst.shape[2],  # W direction
# )

# lst_resized = zoom(lst, zoom=zoom_factors, order=1)  
# print("LST resized:", lst_resized.shape)

# # Combine the tensors
# combined = np.concatenate([sentinel, lst_resized], axis=0)
# print("Combined tensor shape:", combined.shape)
