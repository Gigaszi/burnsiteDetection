from rasterio.plot import plotting_extent

from utils import get_paths_to_bands, combine_tifs, calculate_nbr, plot_nbr, plot_rgb, plot_satellite_imagery, plot_dnbr

all_landsat_bands_path = get_paths_to_bands("bc_burn", [5, 6, 7])
all_landsat_bands_path.sort()

landsat_post_fire = combine_tifs(all_landsat_bands_path)

extent_landsat = plotting_extent(
    landsat_post_fire[0].values, landsat_post_fire.rio.transform())

landsat_postfire_nbr = calculate_nbr(landsat_post_fire)

plot_nbr(landsat_postfire_nbr, extent_landsat)

landsat_postfire_nbr -= 100

plot_dnbr(landsat_postfire_nbr, extent_landsat)