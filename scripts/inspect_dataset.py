import rasterio
import numpy as np

FILE_PATH = r"D:\Projects\Analytica\ssl4eo_l_oli_tirs_toa_benchmark\ssl4eo_l_oli_tirs_toa_benchmark\0000000\LC08_045030_20190814\all_bands.tif"

with rasterio.open(FILE_PATH) as src:

    print("Number of bands:", src.count)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Dtype:", src.dtypes)
    print("NoData:", src.nodata)
    print("Descriptions:", src.descriptions)

    data = src.read()

    print("Shape:", data.shape)
    print("Overall min:", np.nanmin(data))
    print("Overall max:", np.nanmax(data))

    for i in range(src.count):
        band = data[i]

        print(
            f"Band {i+1}: "
            f"min={np.nanmin(band)}, "
            f"max={np.nanmax(band)}, "
            f"mean={np.nanmean(band)}"
        )