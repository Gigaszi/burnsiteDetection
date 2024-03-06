from rasterio.plot import plotting_extent

from utils import (calculate_nbr, plot_nbr,
                   calculate_dnbr, get_pre_and_post_fire_paths, save_dnbr_as_tif_and_hist, calculate_nbr_plus,
                   get_from_config, get_amount_of_pixels_in_classes)


# nbr+
method = "nbr+"
pre_fire, post_fire = get_pre_and_post_fire_paths(get_from_config("satellite")[0], method)

extent_landsat = plotting_extent(
    post_fire[0].values, post_fire.rio.transform())

pre_fire_nbr = calculate_nbr_plus(pre_fire)
post_fire_nbr = calculate_nbr_plus(post_fire)

plot_nbr(pre_fire_nbr, extent_landsat, '28th of April', method)
plot_nbr(post_fire_nbr, extent_landsat, '20th of October', method)

dnbr = calculate_dnbr(pre_fire_nbr, post_fire_nbr)

save_dnbr_as_tif_and_hist(dnbr, extent_landsat, method)

# nbr
method = "nbr"
pre_fire, post_fire = get_pre_and_post_fire_paths(get_from_config("satellite")[0], method)
pre_fire_nbr = calculate_nbr(pre_fire)
post_fire_nbr = calculate_nbr(post_fire)

plot_nbr(pre_fire_nbr, extent_landsat, '28th of April', method)
plot_nbr(post_fire_nbr, extent_landsat, '20th of October', method)

dnbr = calculate_dnbr(pre_fire_nbr, post_fire_nbr)


save_dnbr_as_tif_and_hist(dnbr, extent_landsat, method)

# print(get_amount_of_pixels_in_classes(dnbr))
