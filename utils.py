import os
import re
from glob import glob

import numpy as np
import xarray as xr
import rioxarray as rxr
import yaml
import matplotlib.pyplot as plt
import earthpy.spatial as es
import earthpy.plot as ep
from matplotlib.colors import ListedColormap
from numpy import ma


def get_from_config(key: str):
    with open('config.yaml', 'r') as config_file:
        return yaml.safe_load(config_file)[key]

def combine_tifs(tif_list):
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


def get_paths_to_bands(path_to_directory: str, bands: list):
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


def plot_rgb(image_path, figsize=(10, 10)):
    """
    Plot a three-band image.

    Parameters:
    - image_path (str): The relative path to the image file.
    - figsize (tuple): A tuple of integers representing the figure size.

    Returns:
    - None

    Example:
     plot_rgb('data/example.tif', figsize=(10, 10))

    """

    image = rxr.open_rasterio(image_path, masked=True).squeeze()
    image.plot.imshow(robust=True, figsize=figsize)


def calculate_nbr(bands):
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

def plot_nbr(bands, extent):
    fig, ax = plt.subplots(figsize=(12, 6))

    ep.plot_bands(bands,
                  cmap="viridis",
                  vmin=-1,
                  vmax=1,
                  ax=ax,
                  extent=extent,
                  title="Landsat derived Normalized Burn Ratio\n 23 July 2016 \n Post Cold Springs Fire")

    plt.show()

def plot_satellite_imagery(path_to_directory, bands):

    #
    # stack_band_paths = get_paths_to_bands(path_to_directory, bands)
    # stack_band_paths.sort()
    #
    # arr_st, meta = es.stack(stack_band_paths, nodata=-9999)
    #
    # fig, ax = plt.subplots(figsize=(12, 12))
    #
    # ep.plot_rgb(arr_st,
    #             title='RGB Satellite Image',
    #             stretch=True,  # Apply histogram stretch to improve contrast
    #             str_clip=0.5,  # Adjust this parameter to control the stretch intensity
    #             figsize=(10, 10))
    # plt.show()

    red_band = get_paths_to_bands(path_to_directory, [bands[0]])[0]
    green_band = get_paths_to_bands(path_to_directory, [bands[1]])[0]
    blue_band = get_paths_to_bands(path_to_directory, [bands[2]])[0]

    # Stack the three bands
    arr_st, meta = es.stack([red_band, green_band, blue_band], nodata=-9999)

    fig, ax = plt.subplots(figsize=(12, 12))

    ep.plot_rgb(arr_st,
                title='RGB Satellite Image',
                stretch=True,  # Apply histogram stretch to improve contrast
                str_clip=0.5,  # Adjust this parameter to control the stretch intensity
                figsize=(10, 10))
    plt.show()

def calculate_dnbr(pre_fire_nbr, post_fire_nbr):
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

def plot_dnbr(dnbr, extent):
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



