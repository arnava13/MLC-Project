import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
from scipy.ndimage import zoom
from odc.stac import stac_load

def radiance_to_temperature_landsat(lwir_band):
    K1_CONSTANT = 774.8853
    K2_CONSTANT = 1321.0789
    lwir_band = xr.where(lwir_band <= 0, np.nan, lwir_band)
    temperature = K2_CONSTANT / np.log((K1_CONSTANT / lwir_band) + 1)
    return temperature

def load_sentinel_tensor_from_bbox_median(bounds, time_window, selected_bands=["B02", "B03", "B04", "B08"], resolution_m=10, cloud_cover=30):
    scale = resolution_m / 111320.0
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        bbox=bounds,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": cloud_cover}}
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel data.")
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

    # print("Dimensions before transpose:", arr.dims)  # debug
    arr = arr.transpose("band", "time", "y", "x")
    median_tensor = arr.median(dim="time", skipna=True)
    return median_tensor.transpose("band", "y", "x").values.astype(np.float32)


def load_lst_tensor_from_bbox_median(bounds, time_window, resolution_m=30):
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
        raise ValueError("No Landsat data.")
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
    for i in range(da.sizes["time"]):
        radiance = da.isel(time=i)
        temperature = radiance_to_temperature_landsat(radiance)
        lst_stack.append(temperature.values)
    lst_stack = np.stack(lst_stack, axis=0)
    lst_median = np.nanmedian(lst_stack, axis=0, keepdims=True)
    return lst_median.astype(np.float32)

def create_and_save_cloudless_mosaic(city_name, bounds, output_dir,
                                       time_window: str,
                                       selected_bands=["B02", "B03", "B04", "B08"],
                                       resolution_m=10, cloud_cover=5):
    """Generates and saves a cloudless Sentinel-2 median mosaic for a given time window.

    Args:
        city_name (str): Name of the city.
        bounds (list): Bounding box [min_lon, min_lat, max_lon, max_lat].
        output_dir (Path): Base directory to save the mosaic (within city_name/sat_files).
        time_window (str): Time window in 'YYYY-MM-DD/YYYY-MM-DD' format.
        selected_bands (list): Sentinel-2 bands.
        resolution_m (int): Spatial resolution in meters.
        cloud_cover (int): Cloud cover threshold.

    Returns:
        Path: The path to the saved mosaic file, or None if failed.
    """
    import logging # Add logging within the function if not module-level
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sat_files_dir = Path(output_dir) / city_name / "sat_files"
    sat_files_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on time_window
    start_date_str = time_window.split('/')[0].replace('-', '')
    end_date_str = time_window.split('/')[1].replace('-', '')
    output_filename = f"sentinel_{city_name}_{start_date_str}_to_{end_date_str}_cloudless_mosaic.npy"
    output_path = sat_files_dir / output_filename

    if output_path.exists():
        logging.info(f"Cloudless mosaic {output_path} already exists. Skipping generation.")
        return output_path

    logging.info(f"Generating cloudless mosaic for {city_name}, window {time_window}...")
    try:
        mosaic_data = load_sentinel_tensor_from_bbox_median(
            bounds=bounds,
            time_window=time_window,
            selected_bands=selected_bands,
            resolution_m=resolution_m,
            cloud_cover=cloud_cover
        )
        np.save(output_path, mosaic_data)
        logging.info(f"Saved cloudless mosaic to {output_path}")
        return output_path
    except ValueError as e:
        logging.error(f"No suitable Sentinel data found for mosaic ({city_name}, {time_window}): {e}")
        return None
    except Exception as e:
        logging.error(f"Error generating cloudless mosaic for {city_name}, {time_window}: {e}")
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
