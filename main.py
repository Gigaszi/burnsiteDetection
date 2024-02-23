from rasterio.plot import plotting_extent

from utils import (calculate_nbr, plot_nbr,
                   calculate_dnbr, get_pre_and_post_fire_paths, save_dnbr_as_tif_and_hist, calculate_nbr_plus,
                   get_from_config, get_amount_of_pixels_in_classes)

pre_fire, post_fire = get_pre_and_post_fire_paths(get_from_config("satellite")[0], get_from_config("method")[0])

extent_landsat = plotting_extent(
    post_fire[0].values, post_fire.rio.transform())

if get_from_config("method")[0] == 'nbr':
    pre_fire_nbr = calculate_nbr(pre_fire)
    post_fire_nbr = calculate_nbr(post_fire)
elif get_from_config("method")[0] == 'nbr+':
    pre_fire_nbr = calculate_nbr_plus(pre_fire)
    post_fire_nbr = calculate_nbr_plus(post_fire)

plot_nbr(pre_fire_nbr, extent_landsat, '4th of June')
plot_nbr(post_fire_nbr, extent_landsat, '7th of October')

dnbr = calculate_dnbr(pre_fire_nbr, post_fire_nbr)


save_dnbr_as_tif_and_hist(dnbr, extent_landsat)

print(get_amount_of_pixels_in_classes(dnbr))
