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

def load_sentinel_tensor_from_bbox_median(bounds, time_window, selected_bands=["B02", "B03", "B04", "B08"], resolution_m=10):
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
