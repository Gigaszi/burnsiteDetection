import os
import re
from typing import List

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


def get_paths_to_bands(path_to_directory: str, bands: list, satellite: str) -> list[str] | None:
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
    if satellite == "sentinel":
        if bands is None or len(bands) == 0:
            print("No bands specified.")
            return None
        elif len(bands) == 1:
            bands_pattern = f"B{bands[0]:02d}"
        else:
            bands_pattern = '|'.join(f"B{band}" for band in bands)

        current_directory = os.path.dirname(__file__)
        folder_path = os.path.join(current_directory, path_to_directory)

        pattern = re.compile(rf'.*_(?:{bands_pattern})_.+\.jp2')
        matching_files = [file for file in os.listdir(folder_path) if pattern.match(file)]

        if matching_files:
            return [os.path.join(folder_path, file) for file in matching_files]
        else:
            print("No matching files found.")
            return None
    elif satellite == "landsat":
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
    else:
        print("Satellite not supported.")
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

def calculate_nbr_plus(bands) -> xr.DataArray:
    # ['02', '03', '8A', '12']
    return ((bands[3] - bands[2] - bands[1] - bands[0]) / (bands[3] + bands[2] + bands[1] + bands[0]))

def plot_nbr(bands, extent, date) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ep.plot_bands(bands,
                  cmap='viridis',
                  vmin=-1,
                  vmax=1,
                  ax=ax,
                  extent=extent,
                  title=f"Derived Normalized Burn Ratio\n {date}")

    plt.savefig(f'output/NBR_{date.replace(" ", "_")}.png')


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

def save_dnbr_as_tif_and_hist(dnbr, extent) -> None:
    output_path = get_from_config("output_path")
    dnbr_cat_names = get_from_config("dnbr_cat_names")

    nbr_colors = get_from_config("nbr_colors")
    nbr_cmap = ListedColormap(nbr_colors)

    # Define dNBR classification bins
    # reclassify raster https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/classify-plot-raster-data-in-python/
    dnbr_class_bins = get_from_config("dnbr_class_bins")

    #dnbr_landsat_class = np.digitize(dnbr, dnbr_class_bins)

    dnbr_landsat_class = xr.apply_ufunc(np.digitize,
                                        dnbr,
                                        dnbr_class_bins)

    fig, ax = plt.subplots(figsize=(10, 8))
    dnbr_landsat_class.plot.imshow(cmap=nbr_cmap)
    # Plot the data with a custom legend
    dnbr_landsat_class_plot = ma.masked_array(
        dnbr_landsat_class.values, dnbr_landsat_class.isnull())
    ax.set_title('Difference in NBR+ between 4th of June and 7th of October 2023')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.show()
    plt.savefig(f'{output_path[0]}/classes.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    dnbr_landsat_class.plot()

    # numpy_array = dnbr_landsat_class.values.flatten()
    # plt.bar(range(len(numpy_array)), numpy_array, color=nbr_colors)

    ax.set_title('Difference in NBR+ between 4th of June and 7th of October 2023')
    plt.savefig(f'{output_path[0]}/hist.png')



    classes = np.unique(dnbr_landsat_class_plot)
    classes = classes.tolist()[:5]
    reversed_dnbr = np.flip(dnbr_landsat_class_plot, axis=0)
    transform = from_origin(extent[0], extent[2], dnbr.rio.resolution()[0],
                            dnbr.rio.resolution()[1])


    with rasterio.open(
            output_path[0] + '/dnbr.tif',
            'w',
            driver='GTiff',
            height=dnbr_landsat_class.shape[0],
            width=dnbr_landsat_class.shape[1],
            count=1,
            dtype=str(dnbr_landsat_class.dtype),
            crs=dnbr.rio.crs,
            transform=transform,
    ) as dst:
        dst.write(dnbr_landsat_class, 1)

def get_pre_and_post_fire_paths(satellite, method) -> tuple:
    if satellite == 'sentinel':
        if method == 'nbr':
            bands = ["08", "06", "12"]
        elif method == 'nbr+':
            bands = ['02', '03', '8A', '12']
    elif satellite == 'landsat':
        bands = [5, 6, 7]
    pre_fire = get_paths_to_bands(get_from_config("pre_fire")[0], bands, get_from_config("satellite")[0])
    post_fire = get_paths_to_bands(get_from_config("post_fire")[0], bands, get_from_config("satellite")[0])
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
