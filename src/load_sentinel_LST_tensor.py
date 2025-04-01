import numpy as np
import pystac_client
import planetary_computer
from odc.stac import stac_load
import xarray as xr
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Function to load Sentinel-2 data and create a tensor
def load_sentinel_tensor_from_bbox(bounds=(-74.01, 40.75, -73.86, 40.88), time_window="2021-06-01/2021-09-01", selected_bands=["B02", "B03", "B04", "B08"], resolution_m=10):
    scale = resolution_m / 111320.0  # Convert to degrees for EPSG:4326

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        bbox=bounds,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 30}}
    )

    items = list(search.items())
    print(f"The number of Sentinel-2 scenes: {len(items)}")
    if not items:
        raise ValueError("No data in the area")

    signed_items = [planetary_computer.sign(item) for item in items]

    ds = stac_load(
        signed_items,
        bands=selected_bands,
        crs="EPSG:4326",
        resolution=scale,
        bbox=bounds,
        chunks={"x": 2048, "y": 2048},
        patch_url=planetary_computer.sign
    )

    arr = ds.to_array()
    if "variable" in arr.dims:
        arr = arr.rename({"variable": "band"})
    if "time" in arr.dims:
        arr = arr.isel(time=0)
    if "latitude" in arr.dims or "longitude" in arr.dims:
        arr = arr.rename({"latitude": "y", "longitude": "x"})

    arr = arr.transpose("band", "y", "x")
    print("Sentinel-2 tensor shape:", arr.shape)
    return arr.values.astype(np.float32)

# Function to convert Landsat's LWIR11 band to temperature (K)
def radiance_to_temperature_landsat(lwir_band):
    """Convert Landsat's thermal infrared band to temperature in Kelvin (K)"""
    K1_CONSTANT = 774.8853
    K2_CONSTANT = 1321.0789
    lwir_band = xr.where(lwir_band <= 0, np.nan, lwir_band)
    temperature = K2_CONSTANT / np.log((K1_CONSTANT / lwir_band) + 1)
    return temperature

# Function to load Landsat LST data and create a tensor
def load_lst_tensor_from_bbox(bounds, time_window, resolution_m=30):
    scale = resolution_m / 111320.0  # Convert to degrees for EPSG:4326
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        bbox=bounds,
        datetime=time_window,
        collections=["landsat-c2-l2"],
        query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}},
    )
    items = list(search.items())
    print(f"The number of Landsat scenes: {len(items)}")
    if not items:
        raise ValueError("No Landsat data found.")

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
    if "time" in da.dims:
        da = da.isel(time=0)
    if "latitude" in da.dims or "longitude" in da.dims:
        da = da.rename({"latitude": "y", "longitude": "x"})

    lst_kelvin = radiance_to_temperature_landsat(da)
    lst_tensor = lst_kelvin.values[np.newaxis, ...].astype(np.float32)
    print("LST tensor shape:", lst_tensor.shape)
    return lst_tensor

# Function to combine Sentinel-2 and LST tensors
def combine_sentinel_and_lst_tensor(bounds, time_window):
    print("Loading Sentinel-2 tensor...")
    sentinel_tensor = load_sentinel_tensor_from_bbox(bounds=bounds, time_window=time_window)
    print("→ Sentinel-2 shape:", sentinel_tensor.shape)

    print("Loading LST tensor...")
    lst_tensor = load_lst_tensor_from_bbox(bounds=bounds, time_window=time_window)
    print("→ LST shape:", lst_tensor.shape)

    # Resize LST tensor to match Sentinel-2 dimensions
    _, h_s, w_s = sentinel_tensor.shape
    _, h_l, w_l = lst_tensor.shape
    zoom_factors = (h_s / h_l, w_s / w_l)

    print(f"Resizing LST: zoom={zoom_factors}")
    lst_resized = zoom(lst_tensor[0], zoom_factors, order=1)  # Linear interpolation
    lst_resized = lst_resized[np.newaxis, :, :]  # Shape=(1, H, W)

    # Combine tensors
    combined = np.concatenate([sentinel_tensor, lst_resized], axis=0)
    print("Combined tensor shape:", combined.shape)
    return combined

# Example usage:
bounds = (-74.01, 40.75, -73.86, 40.88)  # Bounding box for New York City
time_window = "2021-06-01/2021-09-01"  # Time window for data search

try:
    # Combine Sentinel-2 and LST tensors
    combined_tensor = combine_sentinel_and_lst_tensor(bounds, time_window)
    print("Final combined tensor shape:", combined_tensor.shape)
    print("LST (K) range: min =", np.nanmin(combined_tensor[-1]), "max =", np.nanmax(combined_tensor[-1]))

    # Visualize LST
    plt.imshow(combined_tensor[-1], cmap="jet")
    plt.colorbar(label="LST (K)")
    plt.title("Landsat-derived LST")
    plt.show()

except Exception as e:
    print("Error occurred:", e)

