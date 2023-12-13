import os
import re

import xarray as xr
import rioxarray as rxr

import matplotlib.pyplot as plt
import earthpy.plot as ep

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
    >>> get_paths_to_bands('data', [3, 4, 5])
    ['/path/to/data/example_B3.TIF', '/path/to/data/example_B4.TIF', '/path/to/data/example_B5.TIF']

    """

    bands_pattern = ','.join(map(str, bands))
    current_directory = os.path.dirname(__file__)
    print("Current Directory:", current_directory)

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
    >>> plot_rgb('data/example.tif', figsize=(10, 10))

    """

    image = rxr.open_rasterio(image_path, masked=True).squeeze()
    image.plot.imshow(robust=True, figsize=figsize)


def calculate_nbr(bands):
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