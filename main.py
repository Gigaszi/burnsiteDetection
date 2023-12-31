from rasterio.plot import plotting_extent

from utils import (calculate_nbr, plot_nbr,
                   calculate_dnbr, get_pre_and_post_fire_paths, save_dnbr_as_tif)

pre_fire, post_fire = get_pre_and_post_fire_paths()

extent_landsat = plotting_extent(
    post_fire[0].values, post_fire.rio.transform())

pre_fire_nbr = calculate_nbr(pre_fire)
post_fire_nbr = calculate_nbr(post_fire)

plot_nbr(pre_fire_nbr, extent_landsat)
plot_nbr(post_fire_nbr, extent_landsat)

dnbr = calculate_dnbr(pre_fire_nbr, post_fire_nbr)


save_dnbr_as_tif(dnbr, extent_landsat)

