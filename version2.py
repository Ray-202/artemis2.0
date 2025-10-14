import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Path to your DEM file
path = r'/Users/reemyalfaisal/artemis/DM2_final_adj_5mpp_slp.tif'

with rasterio.open(path) as src:
    dem = src.read(1)  # elevation values
    nodata = src.nodata
    transform = src.transform
    crs = src.crs

# Replace nodata values with NaN for clean analysis
if nodata is not None:
    slope = np.where(dem == nodata, np.nan, dem)

# Visualize
# Create safe mask
safe_mask = slope <= 12  # degrees threshold

# Plot both
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(slope, cmap='terrain')
ax[0].set_title("Slope Map (°)")
ax[1].imshow(safe_mask, cmap='gray')
ax[1].set_title("Safe Mask (slope ≤ 12°)")
plt.show()



print("DEM shape:", dem.shape)
print("Min elevation:", np.nanmin(dem))
print("Max elevation:", np.nanmax(dem))
print("Mean elevation:", np.nanmean(dem))
print("Standard deviation:", np.nanstd(dem))


plt.figure(figsize=(8, 6))
plt.hist(dem[~np.isnan(dem)].flatten(), bins=50, color='gray', edgecolor='black')
plt.title('Elevation Distribution (DEM)')
plt.xlabel('Elevation (m)')
plt.ylabel('Pixel Count')
plt.show()


min_val = np.nanmin(dem)
max_val = np.nanmax(dem)

min_pos = np.unravel_index(np.nanargmin(dem), dem.shape)
max_pos = np.unravel_index(np.nanargmax(dem), dem.shape)

print(f"Lowest point: {min_val:.2f} m at {min_pos}")
print(f"Highest point: {max_val:.2f} m at {max_pos}")

plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.scatter(min_pos[1], min_pos[0], color='blue', marker='v', label='Lowest')
plt.scatter(max_pos[1], max_pos[0], color='red', marker='^', label='Highest')
plt.legend()
plt.title('DEM — Highest and Lowest Points')
plt.show()
