
import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
from odc.stac import stac_load

def load_sentinel_landsat_tensor(aoi_geojson=None, time_range=("2021-07-01", "2021-07-31"), selected_bands=["B02", "B03", "B04", "B08"]):
    '''
    Sentinel-2bands + Landsat'LST ([band+1, height, width]) tensor as a function.

    Parameters:
        aoi_geojson (dict): GeoJSON: AOIpolygon（None->NYC default）
        time_range (tuple): (start date, end date) 
        selected_bands (list): Sentinel-2' band list

    Returns:
        numpy.ndarray: [band+1, height, width] tesor
    '''

    # default AOI
    if aoi_geojson is None:
        aoi_geojson = {
            "type": "Polygon",
            "coordinates": [[
                [-74.02, 40.70],
                [-74.02, 40.75],
                [-73.95, 40.75],
                [-73.95, 40.70],
                [-74.02, 40.70]
            ]]
        }

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Sentinel-2
    s2_search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi_geojson,
        datetime=f"{time_range[0]}/{time_range[1]}"
    )
    s2_items = list(s2_search.get_items())
    if not s2_items:
        raise ValueError("Cannot find Sentinel-2 data")

    s2_signed = [planetary_computer.sign(item) for item in s2_items]
    ds_s2 = stac_load(
        s2_signed,
        bands=selected_bands,
        crs="EPSG:4326",
        resolution=10
    )

    # Landsat LST
    lst_search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects=aoi_geojson,
        datetime=f"{time_range[0]}/{time_range[1]}"
    )
    lst_items = list(lst_search.get_items())
    if not lst_items:
        raise ValueError("Cannot find Landsat data")

    lst_signed = [planetary_computer.sign(item) for item in lst_items]
    ds_lst = stac_load(
        lst_signed,
        bands=["ST_B10"],  # LST and
        crs="EPSG:4326",
        resolution=30  # Same resolution as Sentinel-2 ??
    )

    # LST interpolation
    ds_lst_interp = ds_lst.interp_like(ds_s2, method="linear")

    # Make into tensor
    sentinel_tensor = ds_s2.to_array().transpose("band", "y", "x").values
    lst_tensor = ds_lst_interp.to_array().squeeze().values[np.newaxis, ...]  # shape: [1, H, W]

    combined_tensor = np.concatenate([sentinel_tensor, lst_tensor], axis=0)  # shape: [band+1, H, W]
    return combined_tensor
