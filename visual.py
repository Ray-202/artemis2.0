# ===================================
# CELL 1: Load and Align Data
# ===================================
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

SLOPE_PATH = "DM2_final_adj_5mpp_slp.tif"
DEM_PATH = "DM2_final_adj_5mpp_surf (1).tif"

# Load both rasters
with rasterio.open(SLOPE_PATH) as slope_src:
    slope = slope_src.read(1)
    slope_meta = slope_src.meta.copy()

with rasterio.open(DEM_PATH) as dem_src:
    dem = dem_src.read(1)
    dem_meta = dem_src.meta.copy()

# Check alignment and reproject if needed
same_crs = (slope_meta["crs"] == dem_meta["crs"])
same_res = np.allclose(slope_src.res, dem_src.res, atol=1e-6)
same_shape = slope.shape == dem.shape

if not (same_crs and same_res and same_shape):
    print("Aligning DEM to slope grid...")
    with rasterio.open(SLOPE_PATH) as slope_src:
        with rasterio.open(DEM_PATH) as dem_src:
            dem_aligned = np.empty(slope.shape, dtype=np.float32)
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_aligned,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=slope_src.transform,
                dst_crs=slope_src.crs,
                resampling=Resampling.bilinear
            )
    aligned_dem = dem_aligned
else:
    aligned_dem = dem

# Calculate ranges
dem_min, dem_max = np.nanmin(aligned_dem), np.nanmax(aligned_dem)
slope_min, slope_max = np.nanmin(slope), np.nanmax(slope)

print(" Data loaded and aligned!")
print(f"Elevation range: {dem_min:.1f} - {dem_max:.1f} m")
print(f"Slope range: {slope_min:.1f} - {slope_max:.1f}°")


# ===================================
# CELL 2: Plot ELEVATION only
# ===================================
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(dem, cmap='terrain')
ax.set_title(f'ELEVATION MAP\n{dem_min:.1f} - {dem_max:.1f} m', 
             fontsize=14, fontweight='bold')
ax.axis('off')

cbar = plt.colorbar(im, ax=ax, fraction=0.046)
cbar.set_label('Elevation (m)', fontsize=11, fontweight='bold')

plt.tight_layout()
#plt.savefig('elevation_only.png', dpi=300, bbox_inches='tight')
plt.show()



# ===================================
# CELL 3: Plot SLOPE only
# ===================================
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(slope, cmap='inferno')
ax.set_title(f'SLOPE MAP\n{slope_min:.1f} - {slope_max:.1f}°', 
             fontsize=14, fontweight='bold')
ax.axis('off')

cbar = plt.colorbar(im, ax=ax, fraction=0.046)
cbar.set_label('Slope (°)', fontsize=11, fontweight='bold')

plt.tight_layout()
#plt.savefig('slope_only.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================
# CELL 4: Plot OVERLAY
# ===================================
fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(aligned_dem, cmap='terrain', alpha=0.6)
im = ax.imshow(slope, cmap='inferno', alpha=0.5)
ax.set_title('OVERLAY\nElevation + Slope', 
             fontsize=14, fontweight='bold')
ax.axis('off')

cbar = plt.colorbar(im, ax=ax, fraction=0.046)
cbar.set_label('Slope (°)', fontsize=11, fontweight='bold')

plt.tight_layout()
#plt.savefig('overlay.png', dpi=300, bbox_inches='tight')
plt.show()



# ===================================
# CELL 5: 3D Elevation Plot (OPTIONAL)
# ===================================
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

print("Preparing 3D visualization...")

# Downsample for 3D
downsample_factor = 5
dem_3d = aligned_dem[::downsample_factor, ::downsample_factor]

# Create coordinate grids
x = np.arange(0, aligned_dem.shape[1], downsample_factor)[:dem_3d.shape[1]]
y = np.arange(0, aligned_dem.shape[0], downsample_factor)[:dem_3d.shape[0]]
X, Y = np.meshgrid(x, y)

# Create 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, dem_3d, cmap='terrain', 
                       linewidth=0, antialiased=False,
                       alpha=0.95, edgecolor='none', shade=True)

ax.set_title(f'3D Elevation Map\n{dem_min:.1f} - {dem_max:.1f} meters', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('X (pixels)', fontsize=11)
ax.set_ylabel('Y (pixels)', fontsize=11)
ax.set_zlabel('Elevation (m)', fontsize=11)
ax.view_init(elev=35, azim=45)

# Set limits
ax.set_xlim(0, aligned_dem.shape[1])
ax.set_ylim(0, aligned_dem.shape[0])
ax.set_zlim(dem_min, dem_max)

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)
cbar.set_label('Elevation (m)', fontsize=11, fontweight='bold')

ax.mouse_init()

plt.tight_layout()
#plt.savefig('lunar_analysis_3D.png', dpi=200, bbox_inches='tight')
plt.show()

print("✅ Done!")