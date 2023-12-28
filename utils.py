import os
import re

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr
import rioxarray as rxr
import yaml
import matplotlib.pyplot as plt
import earthpy.plot as ep
from matplotlib.colors import ListedColormap
from numpy import ma


def get_from_config(key: str) -> list:
    """
    Retrieve a value from the configuration file.

    Parameters:
    - key (str): The key for the desired value in the configuration file.

    Returns:
    - The value associated with the specified key in the configuration file.

    """
    with open('config.yaml', 'r') as config_file:
        return yaml.safe_load(config_file)[key]

def combine_tifs(tif_list) -> xr.DataArray:
    """
    A function that combines a list of tifs in the same CRS
    and of the same extent into an xarray object

    Parameters
    ----------
    tif_list : list
        A list of paths to the tif files that you wish to combine.

    Returns
    -------
    An xarray object with all of the tif files in the listmerged into
    a single object.

    """

    out_xr = []
    for i, tif_path in enumerate(tif_list):
        out_xr.append(rxr.open_rasterio(tif_path, masked=True).squeeze())
        out_xr[i]["band"] = i+1

    return xr.concat(out_xr, dim="band")


def get_paths_to_bands(path_to_directory: str, bands: list) -> list:
    """
    Get file paths for specific bands within a directory.

    Parameters:
    - path_to_directory (str): The relative path to the target directory.
    - bands (List[int]): A list of integer values representing bands.

    Returns:
    - List[str]: A list of file paths corresponding to the specified bands.
                Returns None if no matching files are found.

    Example:
    get_paths_to_bands('data', [3, 4, 5])
    ['/path/to/data/example_B3.TIF', '/path/to/data/example_B4.TIF', '/path/to/data/example_B5.TIF']

    """
    if bands is None or len(bands) == 0:
        print("No bands specified.")
        return None
    elif len(bands) == 1:
        bands_pattern = str(bands[0])
    else:
        bands_pattern = ','.join(map(str, bands))
    current_directory = os.path.dirname(__file__)

    folder_path = os.path.join(current_directory, path_to_directory)

    pattern = re.compile(rf'.*_B[{bands_pattern}]\.TIF')
    matching_files = [file for file in os.listdir(folder_path) if pattern.match(file)]

    if matching_files:
        return [os.path.join(folder_path, file) for file in matching_files]
    else:
        print("No matching files found.")
        return None


def calculate_nbr(bands) -> xr.DataArray:
    """
    Calculate the Normalized Burn Ratio (NBR) using input bands.

    The Normalized Burn Ratio is computed as (NIR - SWIR) / (NIR + SWIR),
    where NIR is the near-infrared band (bands[0]) and SWIR is the shortwave
    infrared band (bands[2]).

    Parameters:
    - bands (DataArray): A DataArray containing the spectral bands.
                             The order is assumed to be [NIR, ..., SWIR, ...].

    Returns:
    - DataArray: A DataArray with the computed Normalized Burn Ratio (NBR) value.

    """

    return (bands[0] - bands[2]) / (bands[0] + bands[2])

def plot_nbr(bands, extent) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ep.plot_bands(bands,
                  cmap="viridis",
                  vmin=-1,
                  vmax=1,
                  ax=ax,
                  extent=extent,
                  title="Landsat derived Normalized Burn Ratio\n 23 July 2016 \n Post Cold Springs Fire")

    plt.show()


def calculate_dnbr(pre_fire_nbr, post_fire_nbr) -> xr.DataArray:
    """
    Calculate the differenced normalized burn ratio (dNBR) using input bands.

    The differenced normalized burn ratio is computed as (pre-fire NBR - post-fire NBR).

    Parameters:
    - pre_fire_nbr (DataArray): A DataArray containing the pre-fire spectral bands.
    - post_fire_nbr (DataArray): A DataArray containing the post-fire spectral bands.

    Returns:
    - DataArray: A DataArray with the computed differenced normalized burn ratio (dNBR) value.

    """

    return pre_fire_nbr - post_fire_nbr

def save_dnbr_as_tif(dnbr, extent) -> None:
    dnbr_cat_names = get_from_config("dnbr_cat_names")

    nbr_colors = get_from_config("nbr_colors")
    nbr_cmap = ListedColormap(nbr_colors)

    # Define dNBR classification bins
    # reclassify raster https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/classify-plot-raster-data-in-python/
    dnbr_class_bins = get_from_config("dnbr_class_bins")

    print(dnbr_class_bins)
    print(dnbr_cat_names)
    #dnbr_landsat_class = np.digitize(dnbr, dnbr_class_bins)

    dnbr_landsat_class = xr.apply_ufunc(np.digitize,
                                        dnbr,
                                        dnbr_class_bins)
    # Plot the data with a custom legend
    dnbr_landsat_class_plot = ma.masked_array(
        dnbr_landsat_class.values, dnbr_landsat_class.isnull())

    fig, ax = plt.subplots(figsize=(10, 8))

    classes = np.unique(dnbr_landsat_class_plot)
    classes = classes.tolist()[:5]
    reversed_dnbr = np.flip(dnbr, axis=0)
    transform = from_origin(extent[0], extent[2], dnbr.rio.resolution()[0],
                            dnbr.rio.resolution()[1])

    output_path = get_from_config("output_path")

    with rasterio.open(
        output_path[0] + '/dnbr.tif',
        'w',
        driver='GTiff',
        height=reversed_dnbr.shape[0],
        width=reversed_dnbr.shape[1],
        count=1,
        dtype=str(reversed_dnbr.dtype),
        crs=dnbr.rio.crs,
        transform=transform,
    ) as dst:
        dst.write(reversed_dnbr, 1)

def get_pre_and_post_fire_paths() -> tuple:
    pre_fire = get_paths_to_bands(get_from_config("pre_fire")[0], [5, 6, 7])
    post_fire = get_paths_to_bands(get_from_config("post_fire")[0], [5, 6, 7])
    pre_fire.sort()
    post_fire.sort()
    pre_fire = combine_tifs(pre_fire)
    post_fire = combine_tifs(post_fire)

    return pre_fire, post_fire

def plot_dnbr(dnbr, extent) -> None:
    dnbr_cat_names = get_from_config("dnbr_cat_names")

    nbr_colors = get_from_config("nbr_colors")
    nbr_cmap = ListedColormap(nbr_colors)

    # Define dNBR classification bins
    # reclassify raster https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/classify-plot-raster-data-in-python/
    dnbr_class_bins = get_from_config("dnbr_class_bins")

    print(dnbr_class_bins)
    print(dnbr_cat_names)
    #dnbr_landsat_class = np.digitize(dnbr, dnbr_class_bins)

    dnbr_landsat_class = xr.apply_ufunc(np.digitize,
                                        dnbr,
                                        dnbr_class_bins)
    # Plot the data with a custom legend
    dnbr_landsat_class_plot = ma.masked_array(
        dnbr_landsat_class.values, dnbr_landsat_class.isnull())

    fig, ax = plt.subplots(figsize=(10, 8))

    classes = np.unique(dnbr_landsat_class_plot)
    classes = classes.tolist()[:5]

    ep.plot_bands(dnbr_landsat_class_plot,
                  cmap=nbr_cmap,
                  vmin=1,
                  vmax=5,
                  title="Landsat dNBR - Cold Spring Fire Site \n June 22, 2016 - July 24, 2016",
                  cbar=False,
                  scale=False,
                  extent=extent,
                  ax=ax)

    ep.draw_legend(im_ax=ax.get_images()[0],
                   classes=classes,
                   titles=dnbr_cat_names)

    plt.show()
