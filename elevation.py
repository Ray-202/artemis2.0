import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION
# ===============================
SLOPE_PATH = "slope_map.tif"     # your slope raster (in degrees)
DEM_PATH   = "ldem_map.tif"      # your LDEM raster (elevation in meters)
ALIGNED_DEM_PATH = "ldem_resampled_to_slope.tif"  # output file
SHOW_PLOTS = True

# ===============================
# STEP 1 — Load both rasters
# ===============================
with rasterio.open(SLOPE_PATH) as slope_src:
    slope = slope_src.read(1)
    slope_meta = slope_src.meta.copy()

with rasterio.open(DEM_PATH) as dem_src:
    dem = dem_src.read(1)
    dem_meta = dem_src.meta.copy()

print("\n--- SLOPE RASTER INFO ---")
print(f"CRS: {slope_meta['crs']}")
print(f"Resolution: {slope_src.res}")
print(f"Shape: {slope.shape}")
print(f"Bounds: {slope_src.bounds}")

print("\n--- DEM RASTER INFO ---")
print(f"CRS: {dem_meta['crs']}")
print(f"Resolution: {dem_src.res}")
print(f"Shape: {dem.shape}")
print(f"Bounds: {dem_src.bounds}")

# ===============================
# STEP 2 — Check if CRS/resolution match
# ===============================
same_crs = (slope_meta["crs"] == dem_meta["crs"])
same_res = np.allclose(slope_src.res, dem_src.res, atol=1e-6)
same_shape = slope.shape == dem.shape

print("\n--- CHECK ALIGNMENT ---")
print(f"Same CRS: {same_crs}")
print(f"Same resolution: {same_res}")
print(f"Same shape: {same_shape}")

# ===============================
# STEP 3 — If not aligned, reproject DEM
# ===============================
if not (same_crs and same_res and same_shape):
    print("\nResampling DEM to match slope grid...")

    transform, width, height = calculate_default_transform(
        dem_src.crs, slope_src.crs,
        slope_src.width, slope_src.height,
        *slope_src.bounds
    )

    dem_aligned = np.empty((height, width), dtype=np.float32)

    reproject(
        source=rasterio.band(dem_src, 1),
        destination=dem_aligned,
        src_transform=dem_src.transform,
        src_crs=dem_src.crs,
        dst_transform=slope_src.transform,
        dst_crs=slope_src.crs,
        resampling=Resampling.bilinear
    )

    # update metadata
    dem_meta.update({
        "driver": "GTiff",
        "height": slope_src.height,
        "width": slope_src.width,
        "transform": slope_src.transform,
        "crs": slope_src.crs
    })

    # write aligned DEM
    with rasterio.open(ALIGNED_DEM_PATH, "w", **dem_meta) as dst:
        dst.write(dem_aligned, 1)

    print(f"Aligned DEM saved to: {ALIGNED_DEM_PATH}")
    aligned_dem = dem_aligned

else:
    print("DEM already aligned with slope.")
    aligned_dem = dem

# ===============================
# STEP 4 — Visualization checks
# ===============================
if SHOW_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(slope, cmap="terrain")
    axes[0].set_title("Slope Map (degrees)")
    axes[0].axis("off")

    axes[1].imshow(aligned_dem, cmap="gray")
    axes[1].set_title("Elevation (LDEM, meters)")
    axes[1].axis("off")

    axes[2].imshow(aligned_dem, cmap="gray", alpha=0.6)
    axes[2].imshow(slope, cmap="terrain", alpha=0.6)
    axes[2].set_title("Overlay: DEM + Slope")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()