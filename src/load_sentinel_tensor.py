import numpy as np
import pystac_client
import planetary_computer
from odc.stac import stac_load

def load_sentinel_tensor_from_bbox(
    bounds=( -74.01, 40.75, -73.86, 40.88 ),
    time_window="2021-06-01/2021-09-01",
    selected_bands=["B02", "B03", "B04", "B08"],
    resolution_m=10
):
    scale = resolution_m / 111320.0  # EPSG:4326 

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        bbox=bounds,
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 30}}
    )

    items = list(search.items())
    print(f"The number of scenes: {len(items)}")
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
    return arr.values.astype(np.float32)


tensor = load_sentinel_tensor_from_bbox()
print("tensor shape:", tensor.shape)
print("min:", np.nanmin(tensor), "max:", np.nanmax(tensor))

