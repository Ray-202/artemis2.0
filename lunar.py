import rasterio

from pyproj import CRS, Transformer

import numpy as np, rasterio

path = r'/Users/reemyalfaisal/artemis/DM2_final_adj_5mpp_slp.tif'

 

# A) You have  pixel indices (row, col)

with rasterio.open(path) as src:

    slope = src.read(1)

    nodata = src.nodata

 

def slope_at_rc(r, c):

    if r < 0 or c < 0 or r >= slope.shape[0] or c >= slope.shape[1]:

        return np.nan

    val = slope[r, c]

    return np.nan if (nodata is not None and val == nodata) else float(val)

 

# B) You have map x,y in the raster CRS(south-polar stereographic, meters)

# • Nearest neighbor:

with rasterio.open(path) as src:

    def slope_at_xy(x, y):

        r, c = src.index(x, y)       # pixel index

        val = src.read(1)[r, c]

        return np.nan if (src.nodata is not None and val == src.nodata) else float(val)

# • Bilinear (interpolated):

with rasterio.open(path) as src:

    # rasterio.sample expects [(x,y), ...] in the dataset CRS

    def slope_at_xy_bilinear(x, y):

        vals = list(src.sample([(x, y)], indexes=1, masked=True))  # bilinear by default

        v = vals[0]

        return np.nan if np.ma.is_masked(v) else float(v)

   
print("Min:", np.nanmin(slope))
print("Max:", np.nanmax(slope))
print("Mean:", np.nanmean(slope))

# C) You have lunar lon,lat (degrees) and need to query slope

# Transform lon,lat → raster CRS (x,y), then use B.

path = r'/Users/reemyalfaisal/artemis/DM2_final_adj_5mpp_slp.tif'

with rasterio.open(path) as src:

    raster_crs = src.crs

    # Source lon/lat CRS for Moon: longlat on sphere of radius 1,737,400 m

    moon_lonlat = CRS.from_proj4('+proj=longlat +R=1737400 +no_defs')

    to_raster = Transformer.from_crs(moon_lonlat, raster_crs, always_xy=True)

 

    def slope_at_lonlat(lon, lat, bilinear=True):

        x, y = to_raster.transform(lon, lat)

        if bilinear:

            vals = list(src.sample([(x, y)], indexes=1, masked=True))

            v = vals[0]

            return np.nan if np.ma.is_masked(v) else float(v)

        else:

            r, c = src.index(x, y)

            v = src.read(1)[r, c]

            return np.nan if (src.nodata is not None and v == src.nodata) else float(v)





import rasterio

import matplotlib.pyplot as plt

import numpy as np

from pyproj import CRS, Transformer

 

path = r'/Users/reemyalfaisal/artemis/DM2_final_adj_5mpp_slp.tif'

src = rasterio.open(path)

slope = src.read(1)

tr = src.transform

 

fig, ax = plt.subplots(figsize=(8, 6))

img = ax.imshow(slope, cmap='terrain')

plt.colorbar(img, ax=ax, label='Slope (°)')

plt.title('DM2 Slope Map (click anywhere)')

plt.xlabel('Pixel column')

plt.ylabel('Pixel row')

 

def onclick(event):

    if not event.inaxes:

        return

    col = int(event.xdata)

    row = int(event.ydata)

    x, y = src.xy(row, col)           # map coordinates (m)

    val = slope[row, col]             # slope (°)

    moon_lonlat = CRS.from_proj4('+proj=longlat +R=1737400 +no_defs')

    tf = Transformer.from_crs(src.crs, moon_lonlat, always_xy=True)

    lon, lat = tf.transform(x, y)

    print(f'Row={row}, Col={col}')

    print(f'Map X={x:.1f}, Y={y:.1f} m')

    print(f'Slope={val:.2f}°')

    print(f'Lon={lon:.4f}°, Lat={lat:.4f}°')

    ax.plot(col, row, 'ro', markersize=6)

    ax.text(col+20, row, f'{val:.1f}°', color='white',

            bbox=dict(facecolor='black', alpha=0.6, pad=2))

    ax.text(col+20, row+200, f'Lon={lon:.4f}°, Lat={lat:.4f}°', color='white',

            bbox=dict(facecolor='black', alpha=0.6, pad=2))

    fig.canvas.draw()

   

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

print("Min:", np.nanmin(slope))
print("Max:", np.nanmax(slope))
print("Mean:", np.nanmean(slope))
