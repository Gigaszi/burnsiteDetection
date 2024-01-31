import os

import rasterio
from rasterio.enums import Resampling

def resample_band(input_path, output_path, target_resolution):

    with rasterio.open(input_path) as src:
        band_data = src.read(1)

        transform = src.transform
        metadata = src.meta.copy()

        resample_scale = target_resolution / transform.a

        transform = rasterio.Affine(transform.a * resample_scale, transform.b, transform.c,
                                    transform.d, transform.e * resample_scale, transform.f)

        metadata['transform'] = transform
        metadata['width'] = src.width // 2
        metadata['height'] = src.height // 2

        resampled_data = band_data.astype(rasterio.uint16)
        resampled_data = resampled_data[::2, ::2]
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(resampled_data, 1)

    os.remove(input_path)

# Example usage
input_band_path = "sentinel2/T10UEA_20231007T191259_B08_10m.jp2"
output_band_path = "sentinel2/T10UEA_20231007T191259_B08_20m.jp2"
target_resolution = 20

resample_band(input_band_path, output_band_path, target_resolution)
